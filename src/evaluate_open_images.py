"""Avalia o peso existente em fotos externas de galinhas, sem retreinar."""
import argparse
import csv
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

from train import detection_metrics, evaluate, load_rows, make_dataset


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/open_images"
CLASS = "/m/09b5t"  # Chicken no Open Images
LICENSE = "https://creativecommons.org/licenses/by/2.0/"
FILES = {
    "validation-annotations-bbox.csv": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "validation-images-with-rotation.csv": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}

# Seleção visual feita antes da inferência: aves vivas e corpo visível.
# Categorias de direção do enquadramento, não de altura física da câmera.
# "lateral" = perfil predominante; "frontal_3_4" = frente ou três quartos.
VIEWS = {
    "01463f5494340d3d": "lateral",
    "0d3da02e79f84dde": "lateral",
    "222c38d9991e97e2": "frontal_3_4",
    "2530c0af68019228": "lateral",
    "313a49ce1db7ace7": "lateral",
    "3cbe64aa3e4ee0b7": "lateral",
    "44dba1c853605435": "lateral",
    "48afeccfea976d06": "lateral",
    "4cf84aa31d937833": "lateral",
    "579c78c63faf394a": "frontal_3_4",
    "6eb70509fcd7d06b": "lateral",
    "85f0d599b3e1d8da": "frontal_3_4",
    "8739bda4dd4a1fd5": "frontal_3_4",
    "8c67f6e6c2ca8a6d": "lateral",
    "afe7ee4009697b48": "lateral",
    "e74990a1ca28e590": "frontal_3_4",
    "f0a8b1d9fc15bf40": "frontal_3_4",
    "f0b05ef0bb73ce15": "lateral",
}


def fetch(url, target):
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)


def prepare():
    for name, url in FILES.items():
        fetch(url, RAW / name)
    with (RAW / "validation-annotations-bbox.csv").open(newline="", encoding="utf-8") as stream:
        boxes = defaultdict(list)
        for row in csv.DictReader(stream):
            if row["LabelName"] == CLASS and row["ImageID"] in VIEWS:
                if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1":
                    raise ValueError(f"Rótulo de grupo/representação: {row['ImageID']}")
                boxes[row["ImageID"]].append(row)
    with (RAW / "validation-images-with-rotation.csv").open(newline="", encoding="utf-8") as stream:
        metadata = {row["ImageID"]: row for row in csv.DictReader(stream) if row["ImageID"] in VIEWS}
    if set(boxes) != set(VIEWS) or set(metadata) != set(VIEWS):
        raise ValueError("Faltam caixas ou metadados para a seleção externa")
    manifest = ROOT / "data/open_images_external.csv"
    attribution = ROOT / "data/open_images_attribution.csv"
    with manifest.open("w", newline="", encoding="utf-8") as out, attribution.open("w", newline="", encoding="utf-8") as credits:
        writer = csv.writer(out)
        source = csv.writer(credits)
        writer.writerow(("image", "label", "group", "perspective"))
        source.writerow(("image_id", "perspective", "boxes", "author", "original_page", "license"))
        for image_id, view in VIEWS.items():
            record = metadata[image_id]
            if record["License"] != LICENSE or record["Rotation"] not in ("0", "0.0", ""):
                raise ValueError(f"Licença ou rotação não verificada: {image_id}")
            image = RAW / "images" / f"{image_id}.jpg"
            label = RAW / "labels" / f"{image_id}.txt"
            fetch(f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg", image)
            label.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for box in boxes[image_id]:
                xmin, xmax, ymin, ymax = (float(box[key]) for key in ("XMin", "XMax", "YMin", "YMax"))
                lines.append(f"0 {(xmin + xmax)/2:.8f} {(ymin + ymax)/2:.8f} {xmax - xmin:.8f} {ymax - ymin:.8f}")
            label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            writer.writerow((image.relative_to(ROOT / "data").as_posix(), label.relative_to(ROOT / "data").as_posix(), record["AuthorProfileURL"] or image_id, view))
            source.writerow((image_id, view, len(lines), record["Author"], record["OriginalLandingURL"], record["License"]))
    return manifest, sum(len(items) for items in boxes.values())


def prepare_selection(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    manifest = RAW / "final_manifest.csv"
    count = 0
    with manifest.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(("image", "label", "group", "perspective"))
        for record in records:
            image_id = record["image_id"]
            if record["split"] != "test" or record["license"] != LICENSE:
                raise ValueError(f"Origem ou licença inválida: {image_id}")
            image = RAW / "test_images" / f"{image_id}.jpg"
            label = RAW / "test_labels" / f"{image_id}.txt"
            fetch(f"https://open-images-dataset.s3.amazonaws.com/test/{image_id}.jpg", image)
            lines = []
            for x1, y1, x2, y2 in record["boxes_xyxy"]:
                if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                    raise ValueError(f"Caixa inválida: {image_id}")
                lines.append(f"0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}")
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            writer.writerow((image.relative_to(RAW).as_posix(), label.relative_to(RAW).as_posix(), record["author_profile"] or image_id, record["perspective"]))
            count += len(lines)
    return manifest, count


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/galinhas.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/open_images_external_metrics.json")
    parser.add_argument("--selection", type=Path, help="Seleção fechada da divisão oficial de teste em JSONL")
    parser.add_argument("--iou", type=float, default=0.7, help="Limiar de NMS para a contagem")
    args = parser.parse_args()

    manifest, count = prepare_selection(args.selection) if args.selection else prepare()
    rows = load_rows(manifest)
    processed = ROOT / ("data/processed/open_images_final" if args.selection else "data/processed/open_images_external")
    yaml = make_dataset(rows, {row["group"]: "val" for row in rows}, processed)
    model = YOLO(str(args.model))
    metrics = model.val(data=str(yaml), split="val", project=str(ROOT / "runs"), name="open_images_external", exist_ok=True, verbose=False)
    report = {
        "source": f"Open Images V5 {'test' if args.selection else 'validation'}, Chicken /m/09b5t",
        "scope": "external_final_no_training_or_tuning" if args.selection else "external_development",
        "images": len(rows), "annotated_boxes": count, "confidence_for_count": 0.25, "iou_for_count": args.iou,
        "global_detection": {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "precision": float(metrics.box.mp), "recall": float(metrics.box.mr)},
        "detection_by_perspective": detection_metrics(model, rows, "val", processed),
        "counts": evaluate(model, rows, "val", processed, 0.25, args.iou),
    }
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rows)} imagens, {count} caixas; métricas em {target}")


if __name__ == "__main__":
    main()
