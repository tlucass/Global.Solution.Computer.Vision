"""Acrescenta fotos anotadas do Open Images V5 somente ao treino."""
import argparse
import csv
import json
import urllib.request
from pathlib import Path

from train import split_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=Path("data/manifest_candidate.csv"))
    parser.add_argument("--selection", type=Path, default=Path("data/open_images_training.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/manifest_diverse.csv"))
    args = parser.parse_args()
    with args.base_manifest.open(newline="", encoding="utf-8") as stream:
        base = list(csv.DictReader(stream))
    assignment = split_groups(base, 42)
    rows = []
    for row in base:
        image = (args.base_manifest.parent / row["image"]).resolve()
        label = (args.base_manifest.parent / row["label"]).resolve()
        rows.append({"image": image.relative_to(args.output.parent.resolve()).as_posix(),
                     "label": label.relative_to(args.output.parent.resolve()).as_posix(),
                     "group": row["group"], "perspective": row["perspective"],
                     "split": assignment[row["group"]]})
    selection = [json.loads(line) for line in args.selection.read_text(encoding="utf-8").splitlines()]
    groups = {row["group"] for row in rows}
    for record in selection:
        if record["license"] != "https://creativecommons.org/licenses/by/2.0/":
            raise ValueError(f"Licença não verificada: {record['image_id']}")
        image_id = record["image_id"]
        group = record["author_profile"] or image_id
        if group in groups:
            raise ValueError(f"Autor duplicado entre divisões: {group}")
        groups.add(group)
        image = args.output.parent / "raw/open_images/train_images" / f"{image_id}.jpg"
        label = args.output.parent / "raw/open_images/train_labels" / f"{image_id}.txt"
        if not image.is_file():
            image.parent.mkdir(parents=True, exist_ok=True)
            partial = image.with_suffix(".partial")
            try:
                urllib.request.urlretrieve(f"https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg", partial)
                partial.replace(image)
            finally:
                partial.unlink(missing_ok=True)
        lines = []
        for x1, y1, x2, y2 in record["boxes_xyxy"]:
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"Caixa inválida: {image_id}")
            lines.append(f"0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows.append({"image": image.relative_to(args.output.parent).as_posix(),
                     "label": label.relative_to(args.output.parent).as_posix(),
                     "group": group, "perspective": record["perspective"], "split": "train"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "label", "group", "perspective", "split"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} imagens; {len(selection)} fotos externas apenas no treino; manifesto {args.output}")


if __name__ == "__main__":
    main()
