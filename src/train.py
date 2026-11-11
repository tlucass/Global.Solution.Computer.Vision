"""Treina e avalia um detector de galinhas a partir de um manifesto YOLO."""
import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image", "label", "group", "perspective"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifesto vazio ou sem colunas {sorted(required)}")
    for row in rows:
        if not row["group"] or not row["perspective"]:
            raise ValueError("group e perspective são obrigatórios")
        if row["perspective"].lower() == "global":
            raise ValueError("global é reservado para o agregado de avaliação")
        for field in ("image", "label"):
            row[field] = (path.parent / row[field]).resolve()
            if not row[field].is_file():
                raise FileNotFoundError(row[field])
        for line in row["label"].read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5 or parts[0] != "0":
                raise ValueError(f"Esperada anotação YOLO de uma classe: {row['label']}")
            values = [float(value) for value in parts[1:]]
            if any(not 0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
                raise ValueError(f"Caixa inválida: {row['label']}")
    return rows


def split_groups(rows, seed):
    if all("split" in row and row["split"] for row in rows):
        assignment = {}
        for row in rows:
            if row["split"] not in {"train", "val", "test"}:
                raise ValueError(f"Divisão inválida: {row['split']}")
            group = row["group"]
            if group in assignment and assignment[group] != row["split"]:
                raise ValueError(f"Grupo {group} atravessa divisões")
            assignment[group] = row["split"]
        if {"train", "val", "test"} - set(assignment.values()):
            raise ValueError("Faltam grupos em treino, validação ou teste")
        return assignment
    groups = sorted({row["group"] for row in rows})
    if len(groups) < 3:
        raise ValueError("São necessários ao menos 3 grupos independentes para treino, validação e teste")
    by_view = defaultdict(list)
    for group in groups:
        views = {row["perspective"] for row in rows if row["group"] == group}
        if len(views) != 1:
            raise ValueError(f"Grupo {group} mistura perspectivas; separe grupos de captura")
        by_view[next(iter(views))].append(group)
    if all(len(items) >= 3 for items in by_view.values()):
        assignment = {}
        for view, items in sorted(by_view.items()):
            random.Random(f"{seed}:{view}").shuffle(items)
            assignment[items[0]] = "test"
            assignment[items[1]] = "val"
            assignment.update({item: "train" for item in items[2:]})
        return assignment
    random.Random(seed).shuffle(groups)
    n_test = max(1, round(len(groups) * 0.2))
    n_val = max(1, round(len(groups) * 0.2))
    if n_test + n_val >= len(groups):
        n_test = n_val = 1
    assignment = {group: "test" for group in groups[:n_test]}
    assignment.update({group: "val" for group in groups[n_test:n_test + n_val]})
    assignment.update({group: "train" for group in groups[n_test + n_val:]})
    return assignment


def make_dataset(rows, assignment, root):
    if root.exists():
        shutil.rmtree(root)
    for index, row in enumerate(rows):
        split = assignment[row["group"]]
        stem = f"{index:06d}"
        for kind, source, suffix in (("images", row["image"], row["image"].suffix), ("labels", row["label"], ".txt")):
            target = root / split / kind / f"{stem}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        row["split"] = split
        row["prepared_stem"] = stem
    yaml = root / "dataset.yaml"
    yaml.write_text(f"path: '{root.resolve().as_posix()}'\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: galinha\n", encoding="utf-8")
    return yaml


def evaluate(model, rows, split, root, confidence, iou):
    selected = [row for row in rows if row["split"] == split]
    by_view = defaultdict(list)
    by_view["global"] = list(selected)
    for row in selected:
        by_view[row["perspective"]].append(row)
    result = {}
    for view, subset in by_view.items():
        image_paths = [str(root / split / "images" / f"{row['prepared_stem']}{row['image'].suffix}") for row in subset]
        errors = []
        zone_errors = [[], [], []]
        cluster_errors = []
        examples = []
        for row, image_path in zip(subset, image_paths):
            prediction = model.predict(image_path, conf=confidence, iou=iou, verbose=False)[0]
            truth = len(row["label"].read_text(encoding="utf-8").splitlines())
            found = len(prediction.boxes)
            errors.append(abs(found - truth))
            true_zones = [0, 0, 0]
            for line in row["label"].read_text(encoding="utf-8").splitlines():
                x = float(line.split()[1])
                true_zones[min(2, int(x * 3))] += 1
            predicted_zones = [0, 0, 0]
            for box in prediction.boxes:
                x = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2) / prediction.orig_shape[1]
                predicted_zones[min(2, max(0, int(x * 3)))] += 1
            for index, (actual, detected) in enumerate(zip(true_zones, predicted_zones)):
                zone_errors[index].append(abs(actual - detected))
            def clustered(zones):
                total = sum(zones)
                return total >= 2 and max(zones) / total >= 0.6
            if clustered(true_zones) != clustered(predicted_zones):
                cluster_errors.append({"image": row["image"].name, "annotated_rule": clustered(true_zones), "detected_rule": clustered(predicted_zones)})
            if found != truth and len(examples) < 10:
                examples.append({"image": row["image"].name, "annotated": truth, "detected": found})
        result[view] = {"images": len(subset), "count_mae": sum(errors) / len(errors), "zone_count_mae": {name: sum(zone_errors[index]) / len(subset) for index, name in enumerate(("esquerda", "centro", "direita"))}, "count_errors": examples, "cluster_rule_error_rate": len(cluster_errors) / len(subset), "cluster_rule_errors": cluster_errors[:10]}
    return result


def detection_metrics(model, rows, split, root):
    result = {}
    perspectives = sorted({row["perspective"] for row in rows if row["split"] == split})
    for index, perspective in enumerate(perspectives):
        subset_root = root / f"eval_{split}_{index}"
        for row in rows:
            if row["split"] != split or row["perspective"] != perspective:
                continue
            stem = row["prepared_stem"]
            for kind, suffix in (("images", row["image"].suffix), ("labels", ".txt")):
                source = root / split / kind / f"{stem}{suffix}"
                target = subset_root / "images" / source.name if kind == "images" else subset_root / "labels" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        yaml = subset_root / "dataset.yaml"
        yaml.write_text(f"path: '{subset_root.as_posix()}'\ntrain: images\nval: images\nnames:\n  0: galinha\n", encoding="utf-8")
        metric = model.val(data=str(yaml), split="val", project="runs", name=f"eval_{split}_perspective_{index}", exist_ok=True, verbose=False)
        result[perspective] = {"map50": float(metric.box.map50), "map50_95": float(metric.box.map), "precision": float(metric.box.mp), "recall": float(metric.box.mr)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--output-model", type=Path, default=Path("models/galinhas.pt"))
    parser.add_argument("--output-metrics", type=Path, default=Path("docs/baseline_metrics.json"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--evaluate-only", action="store_true", help="Avalia um peso existente sem treinar")
    parser.add_argument("--final-test", action="store_true", help="Avalia o teste reservado apenas na rodada final")
    args = parser.parse_args()
    if args.epochs < 1 or not 0 < args.confidence < 1:
        parser.error("epochs deve ser positivo e confidence deve estar entre 0 e 1")
    from ultralytics import YOLO
    rows = load_rows(args.manifest.resolve())
    assignment = split_groups(rows, args.seed)
    coverage = {split: sorted({row["perspective"] for row in rows if assignment[row["group"]] == split}) for split in ("train", "val", "test")}
    print(f"Cobertura de perspectivas por divisão: {coverage}")
    root = args.processed.resolve()
    yaml = make_dataset(rows, assignment, root)
    model = YOLO(args.model)
    if not args.evaluate_only:
        model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=0, seed=args.seed, project="runs", name=args.run_name, exist_ok=True)
        best = Path(model.trainer.best)
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, args.output_model)
        model = YOLO(str(args.output_model))
    report = {"seed": args.seed, "groups": assignment, "perspectives_by_split": coverage, "confidence": args.confidence, "iou": args.iou, "splits": {}}
    for split in (("val", "test") if args.final_test else ("val",)):
        metrics = model.val(data=str(yaml), split=split, project="runs", name=f"eval_{split}", exist_ok=True, verbose=False)
        report["splits"][split] = {
            "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "counts": evaluate(model, rows, split, root, args.confidence, args.iou),
            "detection_by_perspective": detection_metrics(model, rows, split, root),
        }
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modelo: {args.model if args.evaluate_only else args.output_model}; métricas: {args.output_metrics}")


if __name__ == "__main__":
    main()
