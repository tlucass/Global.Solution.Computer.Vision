"""Avalia o indicador visual contra rótulos visuais exploratórios por IA."""
import csv
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from infer import counts


def main():
    model = YOLO("models/galinhas.pt")
    with Path("data/agglomeration_labels.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    evaluated = []
    for row in rows:
        path = Path("data/raw/big_broiler") / row["image"]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        prediction = model.predict(image, conf=0.25, iou=0.5, verbose=False)[0]
        predicted = counts(prediction.boxes, image.shape[1], image.shape[0], 0.6)[-1]
        evaluated.append({"image": row["image"], "perspective": row["perspective"], "visual_label": int(row["visual_cluster"]), "rule_prediction": predicted, "correct": predicted == int(row["visual_cluster"])})
    report = {"model": "models/galinhas.pt", "confidence": 0.25, "nms_iou": 0.5, "criterion": "Revisão visual independente das caixas: maioria das aves visíveis em um grupo com contato ou proximidade de até um comprimento de ave. Rotulagem visual por Codex (IA), sem especialista humano; amostra escolhida por casos claros.", "samples": len(evaluated), "accuracy": sum(item["correct"] for item in evaluated) / len(evaluated), "by_perspective": {}, "errors": [item for item in evaluated if not item["correct"]]}
    for view in sorted({item["perspective"] for item in evaluated}):
        subset = [item for item in evaluated if item["perspective"] == view]
        report["by_perspective"][view] = {"samples": len(subset), "accuracy": sum(item["correct"] for item in subset) / len(subset)}
    Path("docs/agglomeration_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": report["accuracy"], "by_perspective": report["by_perspective"], "errors": len(report["errors"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
