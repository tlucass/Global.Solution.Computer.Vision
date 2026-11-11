"""Processa imagens e vídeos de input/ e substitui os resultados em output/."""
import argparse
import csv
import os
from pathlib import Path
from uuid import uuid4

IMAGE = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO = {".mp4", ".avi", ".mov"}


def counts(boxes, width, height, threshold):
    zones = {"esquerda": 0, "centro": 0, "direita": 0}
    for box in boxes:
        x = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2) / width
        zone = "esquerda" if x < 1 / 3 else "centro" if x < 2 / 3 else "direita"
        zones[zone] += 1
    total = sum(zones.values())
    return [total, *zones.values(), int(total >= 2 and max(zones.values()) / total >= threshold)]


def annotated_frame(prediction, values):
    import cv2

    image = prediction.plot(labels=False, conf=False)
    height, width = image.shape[:2]
    total, left, center, right, clustered = values
    if width < 400 or height < 240:
        message = f"N{total} E{left} C{center} D{right} Agl {'SIM' if clustered else 'NAO'}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_width = cv2.getTextSize(message, font, 1, 1)[0][0]
        scale = min(0.65, (width - 8) / max(1, text_width))
        text_height = cv2.getTextSize(message, font, scale, 1)[0][1]
        panel_height = text_height + 8
        cv2.rectangle(image, (0, 0), (width, panel_height), (20, 20, 20), -1)
        cv2.putText(image, message, (4, text_height + 3), font, scale,
                    (80, 220, 80) if clustered else (80, 190, 255), 1)
        for x in (width // 3, 2 * width // 3):
            cv2.line(image, (x, panel_height), (x, height - 1), (0, 220, 220), 1)
        return image
    scale = min(1.0, max(0.55, width / 900))
    line_height = int(28 * scale) + 6
    panel_height = 2 * line_height + 8
    for x in (width // 3, 2 * width // 3):
        cv2.line(image, (x, panel_height), (x, height - 1), (0, 220, 220), 2)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    cv2.putText(image, f"Total {total} | Esq {left} Centro {center} Dir {right}",
                (8, line_height), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    cv2.putText(image, f"Aglomeracao visual: {'SIM' if clustered else 'NAO'}",
                (8, 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (80, 220, 80) if clustered else (80, 190, 255), 2)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/galinhas.pt"))
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--cluster-share", type=float, default=0.6)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Modelo não encontrado: {args.model}")
    if not 0 < args.cluster_share <= 1:
        parser.error("cluster-share deve estar entre 0 e 1")
    if args.input.resolve() == args.output.resolve():
        parser.error("input e output devem ser diretórios diferentes")
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE | VIDEO) if args.input.is_dir() else []
    if not files:
        parser.error("Nenhuma imagem ou vídeo compatível em input/")
    for source in files:
        destination = args.output / source.name
        summary = args.output / f"{source.name}.csv"
        if source.resolve() == destination.resolve() or source.resolve() == summary.resolve():
            parser.error(f"Saída coincide com entrada: {source}")
        token = uuid4().hex
        temp = args.output / f".{token}{source.suffix}"
        temp_csv = args.output / f".{token}.csv"
        try:
            with temp_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["frame", "total", "esquerda", "centro", "direita", "aglomeração_visual"])
                if source.suffix.lower() in IMAGE:
                    image = cv2.imread(str(source))
                    if image is None:
                        raise ValueError("imagem inválida")
                    prediction = model.predict(image, conf=args.confidence, iou=args.iou, verbose=False)[0]
                    values = counts(prediction.boxes, image.shape[1], image.shape[0], args.cluster_share)
                    writer.writerow([0, *values])
                    if not cv2.imwrite(str(temp), annotated_frame(prediction, values)):
                        raise OSError("falha ao gravar imagem")
                else:
                    capture = cv2.VideoCapture(str(source))
                    if not capture.isOpened():
                        raise ValueError("vídeo inválido")
                    fps = capture.get(cv2.CAP_PROP_FPS) or 25
                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    codec = cv2.VideoWriter_fourcc(*("XVID" if source.suffix.lower() == ".avi" else "mp4v"))
                    video = cv2.VideoWriter(str(temp), codec, fps, (width, height))
                    if not video.isOpened():
                        capture.release()
                        raise OSError("falha ao criar vídeo de saída; confira o codec instalado")
                    frames = 0
                    try:
                        while True:
                            ok, frame = capture.read()
                            if not ok:
                                break
                            prediction = model.predict(frame, conf=args.confidence, iou=args.iou, verbose=False)[0]
                            values = counts(prediction.boxes, width, height, args.cluster_share)
                            writer.writerow([frames, *values])
                            video.write(annotated_frame(prediction, values))
                            frames += 1
                    finally:
                        capture.release()
                        video.release()
                    if frames == 0:
                        raise ValueError("vídeo sem quadros legíveis")
            os.replace(temp, destination)
            os.replace(temp_csv, summary)
            print(f"{source} -> {destination}; {summary}")
        except Exception as exc:
            temp.unlink(missing_ok=True)
            temp_csv.unlink(missing_ok=True)
            print(f"Erro em {source}: {exc}")


if __name__ == "__main__":
    main()
