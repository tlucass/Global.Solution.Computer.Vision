# AviRonda - Applied Computer Vision

Detector de galinhas para imagens e vídeos. O programa grava a imagem ou o vídeo com caixas, contagens por três faixas verticais e um CSV por arquivo. A regra de aglomeração é somente um indicador visual; não diagnostica estresse térmico. Os sensores ambientais previstos para o AviRonda são outro componente do projeto; este repositório processa apenas imagens e vídeos.

## Instalar e usar

Requer Python 3.10 ou superior. Após baixar o repositório, na sua raiz, crie um ambiente Python e instale as bibliotecas. A pasta `.venv` é criada por você uma vez; ela não vem no GitHub nem é criada automaticamente ao executar o detector.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Linux/macOS:**

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

O modelo customizado já acompanha o repositório em `models/galinhas.pt`: não é necessário baixar os dados de treino para fazer inferência. Coloque arquivos `.jpg`, `.jpeg`, `.png`, `.webp`, `.mp4`, `.avi` ou `.mov` em `input/`. A pasta começa vazia. Para um teste rápido, copie as cinco amostras de [`examples/`](examples/) para `input/`, ou então, insira imagens ou vídeos que desejar. Depois execute **um** dos comandos:

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe src\infer.py
```

```bash
# Linux/macOS
./.venv/bin/python src/infer.py
```

Cada entrada produz um arquivo visual com o mesmo nome e um CSV `<nome original>.csv` em `output/`. Cada linha do CSV informa quadro, total, contagem à esquerda/centro/direita e `aglomeração_visual`. Executar de novo substitui os resultados de mesmo nome sem modificar as entradas. Vídeos `.mov` dependem do codec disponível no OpenCV. As três faixas têm larguras iguais; a zona vem do centro da caixa. Aglomeração visual vale 1 quando uma faixa reúne ao menos 60% das detecções e há duas ou mais detecções.

As amostras em `examples/` são apenas entradas para experimentar o comando, sem caixas de gabarito. `big_broiler_obliqua.jpg` e `big_broiler_superior.jpg` são quadros dos vídeos 049 e 086 do [Big Broiler Dataset](https://github.com/decide-ugent/big-broiler-dataset), de Cardoen e colaboradores (CV4Animals 2025), sob AGPL-3.0. Os dois `.mp4` são trechos de 12 quadros desses mesmos vídeos, reduzidos a 640 × 360 pixels, sem anotações. `open_images_lateral.jpg` reproduz a fotografia [“Roosters on Duval St.”](https://www.flickr.com/photos/shawnzlea/439326603), de Shawn Rossi, disponível sob [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) e selecionada do Open Images V5. As amostras não representam todas as condições de um aviário.

## Resultados e limites

O treino combina quadros oblíquos e superiores do [Big Broiler](https://github.com/decide-ugent/big-broiler-dataset), fotos laterais/frontais do [Open Images V5](https://storage.googleapis.com/openimages/web/download_v5.html) e quadros superiores densos do [PIO](https://zenodo.org/records/16686320). A escolha do peso e dos limiares usou validação; os testes foram mantidos separados por vídeo, semana ou autor. Métricas detalhadas e erros estão em [experimentos](docs/experiments.md), [teste Big Broiler/PIO](docs/final_metrics.json) e [teste externo Open Images](docs/open_images_final_round2_metrics.json).

| Teste anotado | Imagens | mAP@0,5 | mAP@0,5:0,95 | MAE de contagem |
|---|---:|---:|---:|---:|
| Big Broiler oblíqua | 20 | 0,989 | 0,67 | 0,60 |
| Big Broiler superior | 20 | 0,822 | 0,63 | 2,05 |
| PIO superior densa | 20 | 0,858 | 0,50 | 7,75 |
| Open Images lateral | 24 | 0,719 | 0,413 | 0,458 |
| Open Images frontal/três quartos | 6 | 0,816 | 0,479 | 0,667 |

O teste Big Broiler já tinha sido visto no baseline e não comprova transferência para outro aviário. O PIO compartilha câmera/local entre semanas. As fotos externas são poucas e podem ter aves sem anotação. Uma panorâmica escura, oblíqua e muito densa colocada em `input/` teve grande subcontagem e uma caixa sobre várias aves; as imagens de `input/` não têm gabarito completo. Não use a contagem ou o indicador de aglomeração dessas cenas para decisões de manejo. A medição ambiental na altura das aves é uma proposta dos **sensores** do AviRonda; o detector de imagens ainda não foi avaliado com gabarito de uma **câmera** instalada nessa altura dentro do aviário.

## Reproduzir o treino

Os conjuntos de treino completos não acompanham o repositório. A [lista de fontes, licenças e divisões](data/README.md) descreve o que cada preparação baixa. Esta seção é necessária apenas para reproduzir o treino, não para usar o modelo pronto. O PIO exige aproximadamente 1,5 GB de download e um `tar` com suporte a RAR. Nos comandos abaixo, `python` significa o Python da `.venv`: no Windows PowerShell, use `.\.venv\Scripts\python.exe` no lugar de `python`; em Linux/macOS, use `./.venv/bin/python`. Execute na raiz do projeto:

```bash
python src/prepare_big_broiler.py --train-frames 200 --manifest data/manifest_candidate.csv
python src/prepare_open_images.py
python src/prepare_pio.py
python src/train.py --manifest data/manifest_diverse.csv --model yolo11n.pt --epochs 12 --imgsz 512 --batch 4 --seed 42 --iou 0.5 --processed data/processed_diverse --run-name diverse_v1 --output-model models/galinhas_diverse.pt --output-metrics data/raw/diverse_val_metrics.json
python src/train.py --manifest data/manifest_pio.csv --model models/galinhas_diverse.pt --epochs 5 --imgsz 512 --batch 4 --seed 42 --iou 0.5 --processed data/processed_pio --run-name pio_v1 --output-model models/galinhas.pt --output-metrics data/raw/pio_val_metrics.json
```

Esses comandos treinam e avaliam a validação. Para reproduzir o teste após fixar o peso, execute `python src/train.py --manifest data/manifest_pio.csv --model models/galinhas.pt --evaluate-only --final-test --iou 0.5 --processed data/processed_pio --output-metrics docs/final_metrics.json` e `python src/evaluate_open_images.py --selection data/open_images_final_round2.jsonl --model models/galinhas.pt --iou 0.5 --output docs/open_images_final_round2_metrics.json`. A reprodução pode variar com hardware e versões das bibliotecas.

## Licença

Código e peso distribuídos sob [AGPL-3.0](LICENSE). Big Broiler é AGPL-3.0 (Cardoen e colaboradores), PIO é CC BY 4.0 e as fotos selecionadas do Open Images têm CC BY 2.0 com caixas CC BY 4.0. Atribuições individuais estão nos arquivos de seleção em `data/`.
