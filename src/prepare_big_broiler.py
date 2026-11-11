"""Baixa uma amostra reproduzível do Big Broiler Dataset e converte caixas para YOLO.

Fonte: https://github.com/decide-ugent/big-broiler-dataset (AGPL-3.0).
"""
import argparse
import csv
import io
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

URL = "https://datasets.ilabt.imec.be/broilertracking/big-broiler_v1.zip"
SIZE = 33304319904
IDS = (49, 50, 51, 86, 87, 88)


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = SIZE - self.position
        if size == 0:
            return b""
        request = urllib.request.Request(URL, headers={"Range": f"bytes={self.position}-{self.position + size - 1}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        self.position += len(data)
        return data


def main():
    import cv2
    from train import split_groups

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-frames", type=int, default=20,
                        help="Quadros por vídeo de treino; validação e teste permanecem com 20")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    args = parser.parse_args()
    if args.train_frames < 20:
        parser.error("train-frames deve ser pelo menos 20")

    raw = Path("data/raw/big_broiler")
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest
    with zipfile.ZipFile(RemoteZip()) as archive:
        index = list(csv.DictReader(io.StringIO(archive.read("big-broiler/dataset_manifest.csv").decode())))
        selected = [row for row in index if int(row["video_id"]) in IDS]
        if len(selected) != len(IDS):
            raise ValueError("Vídeos esperados não encontrados no arquivo original")
        assignment = split_groups([
            {"group": row["stem"], "perspective": "obliqua" if row["setup_id"] == "2" else "superior"}
            for row in selected
        ], 42)
        output = []
        for row in selected:
            stem = row["stem"]
            video_member = archive.getinfo(f"big-broiler/videos/{stem}.mp4")
            annotation_member = archive.getinfo(f"big-broiler/annotations/{stem}.csv")
            video_path = raw / f"{stem}.mp4"
            if not video_path.is_file() or video_path.stat().st_size != video_member.file_size:
                print(f"Baixando vídeo {row['video_id']} ({video_member.file_size / 1e6:.0f} MB)", flush=True)
                with archive.open(video_member) as source, video_path.open("wb") as target:
                    while chunk := source.read(4 * 1024 * 1024):
                        target.write(chunk)
            annotations = defaultdict(list)
            lines = io.StringIO(archive.read(annotation_member).decode("utf-8-sig"))
            for box in csv.DictReader(lines):
                annotations[int(box["frame"])].append(box)
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"Vídeo ilegível: {video_path}")
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            count = args.train_frames if assignment[stem] == "train" else 20
            # Quadros espaçados por vídeo. Não misturar quadros do mesmo vídeo nas divisões.
            for frame_id in [round((i + 1) * frames / (count + 1)) for i in range(count)]:
                image_path = raw / f"{stem}_{frame_id:06d}.jpg"
                label_path = image_path.with_suffix(".txt")
                manifest_row = {"image": image_path.relative_to(manifest_path.parent).as_posix(), "label": label_path.relative_to(manifest_path.parent).as_posix(), "group": stem, "perspective": "obliqua" if row["setup_id"] == "2" else "superior"}
                if image_path.is_file() and label_path.is_file():
                    output.append(manifest_row)
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Quadro {frame_id} ilegível em {video_path}")
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Falha ao salvar {image_path}")
                labels = []
                for box in annotations[frame_id]:
                    x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
                    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                        continue
                    labels.append(f"0 {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
                label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                output.append(manifest_row)
            capture.release()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "label", "group", "perspective"))
        writer.writeheader()
        writer.writerows(output)
    print(f"Preparadas {len(output)} imagens em {manifest_path}")


if __name__ == "__main__":
    main()
