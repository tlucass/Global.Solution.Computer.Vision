"""Prepara a amostra anotada e separada por semana do PIO (CC BY 4.0)."""
import csv
import hashlib
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw/pio"
ARCHIVE = RAW / "data.rar"
URL = "https://zenodo.org/api/records/16686320/files/data.rar/content"
SIZE = 1523257163
MD5 = "c207661257123da12df2ea700fe4c2d1"


def fetch_archive():
    RAW.mkdir(parents=True, exist_ok=True)
    offset = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
    if offset > SIZE:
        raise ValueError("Arquivo PIO maior que o original")
    if offset < SIZE:
        request = urllib.request.Request(URL, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=900) as response:
            if offset and response.status != 206:
                raise RuntimeError("Servidor não retomou o download PIO")
            with ARCHIVE.open("ab" if offset else "wb") as stream:
                while block := response.read(8 * 1024 * 1024):
                    stream.write(block)
    if ARCHIVE.stat().st_size != SIZE:
        raise ValueError("Download PIO incompleto")
    digest = hashlib.md5()
    with ARCHIVE.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != MD5:
        raise ValueError("Checksum PIO diferente do registro Zenodo")


def main():
    fetch_archive()
    with (DATA / "pio_selection.csv").open(newline="", encoding="utf-8") as stream:
        selected = list(csv.DictReader(stream))
    if not selected:
        raise ValueError("Seleção PIO vazia")
    members = []
    for row in selected:
        image = row["archive_image"]
        if not image.startswith("data/images/") or ".." in Path(image).parts or not image.endswith(".jpg"):
            raise ValueError(f"Caminho PIO inválido: {image}")
        label = image.replace("/images/", "/labels/").removesuffix(".jpg") + ".txt"
        members.extend((image, label))
    if len(set(members)) != len(members):
        raise ValueError("Imagem ou rótulo PIO duplicado")
    extracted = RAW / "selected"
    extracted.mkdir(parents=True, exist_ok=True)
    listfile = RAW / "members.txt"
    listfile.write_text("\n".join(members) + "\n", encoding="utf-8")
    subprocess.run(["tar", "-xf", str(ARCHIVE), "-C", str(extracted), "-T", str(listfile)], check=True)
    with (DATA / "manifest_diverse.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    discarded = 0
    for row in selected:
        image_member = row["archive_image"]
        label_member = image_member.replace("/images/", "/labels/").removesuffix(".jpg") + ".txt"
        image = extracted / image_member
        original_label = extracted / label_member
        if not image.is_file() or not original_label.is_file():
            raise FileNotFoundError((image, original_label))
        clean_label = RAW / "clean_labels" / row["split"] / original_label.name
        clean_label.parent.mkdir(parents=True, exist_ok=True)
        valid_lines = []
        for line in original_label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            valid = len(parts) == 5 and parts[0] == "0"
            if valid:
                values = [float(value) for value in parts[1:]]
                valid = all(0 <= value <= 1 for value in values) and values[2] > 0 and values[3] > 0
            if valid:
                valid_lines.append(line)
            else:
                discarded += 1
        clean_label.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")
        rows.append({"image": image.relative_to(DATA).as_posix(),
                     "label": clean_label.relative_to(DATA).as_posix(),
                     "group": row["group"], "perspective": row["perspective"], "split": row["split"]})
    output = DATA / "manifest_pio.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "label", "group", "perspective", "split"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(selected)} imagens PIO; {discarded} caixas degeneradas descartadas; {output}")


if __name__ == "__main__":
    main()
