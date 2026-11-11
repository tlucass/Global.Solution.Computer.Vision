# Experimentos

## Baseline 1 — 10/11/2026

Fonte: [Big Broiler Dataset](https://github.com/decide-ugent/big-broiler-dataset), licença AGPL-3.0. `src/prepare_big_broiler.py` selecionou 20 quadros espaçados de cada um de seis vídeos, total de 120 imagens e 1.076 caixas da classe galinha. A amostra tem 60 imagens oblíquas (setup 2) e 60 superiores (setup 3). Há 20 imagens de cada perspectiva em treino, validação e teste. Vídeos 051/088 ficaram no treino, 049/086 na validação e 050/087 no teste, com semente 42. Quadros de um vídeo não atravessam divisões. Vídeos do mesmo setup podem compartilhar câmera ou local; esta avaliação não mede transferência para outro aviário.

YOLO11n inicializado com `yolo11n.pt`, treinado customizadamente por 10 épocas em CPU, tamanho 416, batch 4, sem busca de hiperparâmetros. Comando:

```bash
python src/prepare_big_broiler.py
python src/train.py --manifest data/manifest.csv --epochs 10 --imgsz 416 --batch 4 --final-test
```

| Divisão/perspectiva | Imagens | mAP@0.5 | mAP@0.5:0.95 | Precisão | Recall | MAE contagem | Erro regra visual |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validação global | 40 | 0,736 | 0,449 | 0,830 | 0,770 | 11,625 | 27,5% |
| Validação oblíqua | 20 | 0,790 | 0,432 | 0,905 | 0,761 | 14,400 | 20,0% |
| Validação superior | 20 | 0,728 | 0,515 | 0,743 | 0,782 | 8,850 | 35,0% |
| Teste global | 40 | 0,863 | 0,532 | 0,838 | 0,859 | 7,100 | 45,0% |
| Teste oblíquo | 20 | 0,952 | 0,544 | 0,888 | 0,891 | 3,950 | 15,0% |
| Teste superior | 20 | 0,802 | 0,557 | 0,782 | 0,889 | 10,250 | 75,0% |

Métricas completas, grupos, erros por zona e exemplos por imagem: [baseline_metrics.json](baseline_metrics.json). A contagem usa confiança 0,25. `cluster_rule_error_rate` compara a regra geométrica aplicada às caixas previstas e anotadas; não valida a relevância zootécnica da regra.

## Erros e limites observados

No teste, a contagem superior erra em média 10,25 aves por imagem, apesar do mAP global alto. A média global esconde esse problema. Na imagem oblíqua `050_2_31_1920_011_1_05_000357.jpg`, a inspeção visual da inferência mostrou caixas duplicadas sobre algumas aves e uma caixa na estrutura do bebedouro. Na amostra superior, apareceram caixas sobre o comedouro e regiões sem aves. Esses são falsos positivos que inflam a contagem. Uma correspondência gulosa entre caixas previstas e anotadas com IoU ≥ 0,5 e confiança ≥ 0,25 e NMS IoU 0,7 encontrou, nas 20 imagens oblíquas de teste, 91 falsos positivos e 14 aves perdidas; nas 20 superiores, 206 falsos positivos e 1 ave perdida. O exemplo oblíquo citado teve 5 falsos positivos e 1 ave perdida. [Exemplos por imagem](error_examples.json). A correspondência é uma análise diagnóstica com limiar fixo, não a definição do mAP.

O conjunto de treino é pequeno e vem de apenas dois setups. Não há imagens laterais ou na altura das aves no treino. Há apenas 12 rótulos visuais exploratórios de aglomeração e nenhuma evidência de estresse térmico. O teste final foi executado uma vez para este baseline; qualquer ajuste futuro de confiança, arquitetura ou dados deve usar apenas validação ou um novo teste externo reservado.

## Avaliação externa — Open Images, 10/11/2026

Antes de executar o modelo, foram fixados 18 IDs de fotografias da divisão de validação do [Open Images V5](https://storage.googleapis.com/openimages/web/download_v5.html), classe Chicken, de autores distintos. Após a primeira execução, a revisão visual corrigiu as categorias de enquadramento para 12 perfis laterais predominantes e 6 frontais/três quartos; IDs, peso e limiares não foram escolhidos com base nas métricas. A classificação de direção do enquadramento é heurística e a altura física da câmera não está documentada. Todas as fotos têm licença CC BY 2.0 registrada nos metadados individuais; [atribuição e página original](../data/open_images_attribution.csv). [As anotações](https://storage.googleapis.com/openimages/web/factsfigures_v5.html#licenses) são CC BY 4.0 (Google LLC). As 31 caixas oficiais foram convertidas sem anotação manual. Fotografias de ave cozida, representações, apenas cabeças e caixas marcadas como grupo foram excluídas da seleção visual. O recorte contém galos/galinhas de ambientes externos, diferente dos frangos brancos densos do treino. Não houve retreino nem escolha de limiar a partir destes resultados. Comando: `python src/evaluate_open_images.py`.

| Vista externa | Imagens | Caixas | mAP@0.5 | mAP@0.5:0.95 | Precisão | Recall | MAE contagem |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global | 18 | 31 | 0,0058 | 0,0024 | 0,0195 | 0,226 | 16,72 |
| Frontal/três quartos | 6 | 18 | 0,0232 | 0,0097 | 0,0479 | 0,389 | 18,50 |
| Lateral | 12 | 13 | 0,0002 | 0,0001 | 0,0008 | 0,231 | 15,83 |

[Métricas e erros por imagem](open_images_external_metrics.json). A contagem usa confiança 0,25 e NMS IoU 0,7 como no baseline; o MAE é relativo às **caixas oficiais**, não uma contagem humana exaustiva. A foto `3cbe64aa3e4ee0b7` contém outra ave parcialmente visível além da única caixa oficial, possível subanotação. Um galo lateral com uma caixa anotada (`01463f5494340d3d`) recebeu sete detecções; outro (`6eb70509fcd7d06b`) recebeu 44. Isso sugere caixas espúrias e fragmentação da ave em múltiplas detecções. O mAP, a precisão e o recall foram calculados pelo Ultralytics neste recorte positivo, não pelo protocolo oficial Open Images. A avaliação externa mostra falha severa de transferência para essas vistas e aparências. Como são apenas 18 fotos selecionadas e não há imagens de aviário lateral, não se deve extrapolar o valor numérico para a operação. A altura aproximada das aves em aviário continua sem avaliação. A regra geométrica de aglomeração não é interpretável nessas fotos individuais; seu baixo erro de concordância com caixas anotadas não valida aglomeração real. As anotações oficiais e a classificação de perspectiva ainda requerem auditoria humana independente.

## Revisão visual exploratória da regra

Após o baseline, foram selecionados 12 quadros do teste (seis por perspectiva) com casos visualmente claros. Codex (IA) atribuiu rótulos visuais **sem consultar as caixas**: `1` quando a maioria das aves visíveis forma um grupo com contato ou distância de até um comprimento de ave; `0` quando há grupos pequenos separados ou aves espalhadas. Os rótulos e justificativas estão em [agglomeration_labels.csv](../data/agglomeration_labels.csv). `python src/evaluate_cluster.py` comparou esses rótulos com o indicador calculado das detecções: 9/12 acertos (75%), sendo 6/6 oblíquos e 3/6 superiores. Os três erros e os nomes das imagens constam em [agglomeration_eval.json](agglomeration_eval.json). A seleção deliberada de casos claros, o mesmo vídeo por vista e a rotulagem por uma única IA limitam fortemente essa estimativa. O critério é visual, não um diagnóstico de estresse térmico ou bem-estar; requer validação por especialista e outros locais.

## Execução da inferência

Foram processados `input/amostra_obliqua.jpg` e `input/amostra_superior.mp4` com o peso customizado. Saíram visual e CSV para cada um. Uma segunda execução manteve os quatro mesmos nomes, atualizou os arquivos e preservou os hashes das duas entradas. Esses arquivos são locais e excluídos do Git.

## Inspeção de imagens adicionais em `input/` — 10/11/2026

O indicador de aglomeração já era gravado no CSV, mas não aparecia sobre a imagem. O comando agora desenha os limites das três zonas e mostra contagens e `Aglomeracao visual: SIM/NAO` em imagens e quadros de vídeo. A imagem `.webp` também é aceita. As prévias criadas apenas para inspecionar o PDF foram removidas de `output/`.

Na amostra oblíqua, o CSV registra 15 detecções (1 esquerda, 12 centro, 2 direita) e indicador `SIM`; a inspeção visual mostra caixas duplicadas e uma caixa sobre a estrutura do bebedouro. Na foto pequena `cama-do-aviario-cursos-cpt.jpg`, o CSV registra 21 detecções (9, 8, 4) e `NAO`; muitas caixas estão em espaços da cerca e até na faixa com texto/logotipo, indicando falsos positivos. Na foto panorâmica `.webp`, aparecem muitas aves sem caixa enquanto o CSV registra apenas 27 detecções (5, 9, 13) e `NAO`. Nesse caso a regra de zonas recebe uma contagem muito incompleta e seu resultado não deve ser interpretado como ausência real de aglomeração.

Essas três fotos não têm caixas de referência exaustivas nem licença documentada para treino. A inspeção é qualitativa: não produz MAE ou mAP nem justifica mudar confiança, arquitetura ou peso com base nelas. Uma melhoria verificável exige fotografias/vídeos de aviário de outras câmeras com caixas completas, revisão de falsos positivos em estruturas e divisão por captura; usar validação para escolher mudanças e reservar novo teste independente para medi-las. A vista lateral e a cena densa precisam de atenção especial. O teste final do baseline não será usado para ajustar parâmetros.

## Rodada 2 — revisão das saídas e mais quadros anotados, 11/11/2026

Revisei as quatro saídas das imagens em `input/` do baseline. `images.jpg` (15 detecções) contém aves sem caixa e caixa sobre estrutura de alimentação. `images (1).jpg` (19) tem caixas sobrepostas na mesma ave e aves perdidas ao fundo. `cama-do-aviario-cursos-cpt.jpg` (21) tem muitas caixas em frestas da cerca, cama vazia e faixa de letras/logotipo. A panorâmica `.webp` (27) tem caixas em objetos/áreas sem ave e muitas aves sem caixa; as minúsculas no fundo não são exigidas na inspeção qualitativa. Essas imagens **não têm gabarito completo**, portanto os totais são apenas saídas do modelo, não medidas de acurácia.

Para corrigir o tamanho limitado do treino, aumentei apenas a extração dos vídeos de **treino** de 20 para 200 quadros espaçados por vídeo. Validação e teste continuaram com 20 quadros por vídeo. A amostra Big Broiler passou a 480 quadros e 4.300 caixas. Treinei YOLO11n por 10 épocas, tamanho 416 e batch 4 com o mesmo seed 42. O limiar de inferência foi selecionado **somente na validação**: 0,25 deu MAE 1,025, enquanto 0,70 deu MAE 2,700. A escolha preserva recall. O treino usou `python src/prepare_big_broiler.py --train-frames 200 --manifest data/manifest_candidate.csv` e `python src/train.py --manifest data/manifest_candidate.csv --epochs 10 --imgsz 416 --batch 4 --confidence 0.7 --output-model models/galinhas_candidate.pt --output-metrics runs/candidate_metrics.json --processed data/processed_candidate --run-name candidate_more_frames`. A contagem da tabela foi reavaliada sem treino com `python src/train.py --manifest data/manifest_candidate.csv --model models/galinhas_candidate.pt --evaluate-only --confidence 0.25 --output-metrics data/raw/candidate_val_metrics_025.json --processed data/processed_candidate`.

| Validação | mAP50 | mAP50:95 | Precisão | Recall | MAE contagem |
|---|---:|---:|---:|---:|---:|
| Global, 40 quadros | 0,837 | 0,582 | 0,875 | 0,842 | 1,025 |
| Oblíqua, 20 quadros | 0,850 | 0,567 | 0,936 | 0,836 | 0,850 |
| Superior, 20 quadros | 0,806 | 0,611 | 0,769 | 0,917 | 1,200 |

Com IoU ≥ 0,5 e confiança 0,25, uma correspondência gulosa nas 356 caixas de validação encontrou 309 verdadeiros positivos, 72 falsos positivos e 47 aves perdidas (precisão diagnóstica 0,811, recall 0,868). A contagem pode ter erro baixo por compensação entre falsos positivos e perdas. A regra de aglomeração discordou das caixas anotadas em 25% dos quadros; isso ainda não valida aglomeração real.

Na avaliação externa Open Images de 18 fotos, sem ajuste usando essas fotos, o modelo atingiu mAP50 global 0,0086; mAP50 lateral 0,0000 e frontal/três quartos 0,0441. Portanto, a meta da validação por vídeo foi atingida, mas **não** há transferência aceitável para fotografias laterais. Uma inferência exploratória nas quatro fotos de `input/` reduziu caixas sobre letras na foto pequena, porém detectou só 2 aves nela e 2 na panorâmica; uma caixa grande abrangeu várias aves em `images.jpg`. Não substituí o peso principal por este candidato.

## Rodada 3 — fotos externas anotadas no treino

Para enfrentar a lacuna lateral/frontal sem usar as 18 fotos externas de avaliação, selecionei visualmente 159 fotos da divisão oficial de **treino** do Open Images V5 (318 caixas de galinha). A seleção registrada em [open_images_training.jsonl](../data/open_images_training.jsonl) traz ID, autor, página original, licença e caixas; cada autor aparece uma vez e nenhum dos autores coincide com a avaliação externa. Rejeitei fotos de representação, somente cabeça, comida, rótulos de grupo e rótulos indefinidos. As fotos incluem vistas laterais e frontais, mas são majoritariamente de criação externa, não imagens de câmera na altura das aves dentro de aviário. O manifesto combinado mantém os mesmos 40 quadros de validação e 40 de teste do Big Broiler; as fotos novas entram somente no treino. A auditoria das folhas de contato não garante rótulos exaustivos.

Com `python src/prepare_open_images.py` foi criado o manifesto combinado de 559 imagens de treino: 200 oblíquas, 200 superiores e 159 fotos externas. Treinei YOLO11n desde `yolo11n.pt` por 12 épocas, tamanho 512, batch 4, semente 42, CPU: `python src/train.py --manifest data/manifest_diverse.csv --model yolo11n.pt --epochs 12 --imgsz 512 --batch 4 --processed data/processed_diverse --run-name diverse_v1 --output-model models/galinhas_diverse.pt --output-metrics data/raw/diverse_val_metrics.json`. O peso foi selecionado pela validação por vídeo e pelas 18 fotos externas **de desenvolvimento**, já examinadas no baseline. Estas 18 fotos deixam de ser um teste independente. A seleção [externa final](../data/open_images_final.jsonl) de 30 fotos e 43 caixas da divisão oficial `test` foi fechada por inspeção antes da primeira execução do novo peso; seus 30 autores não coincidem com os de treino ou desenvolvimento.

| Conjunto / vista | Imagens | mAP50 | mAP50:95 | Precisão | Recall | MAE contagem |
|---|---:|---:|---:|---:|---:|---:|
| Big Broiler val / global | 40 | 0,866 | 0,582 | 0,838 | 0,841 | 0,975 |
| Big Broiler val / oblíqua | 20 | 0,934 | 0,622 | 0,896 | 0,848 | 0,650 |
| Big Broiler val / superior | 20 | 0,748 | 0,517 | 0,764 | 0,859 | 1,300 |
| Open Images desenvolvimento / global | 18 | 0,655 | 0,297 | 0,611 | 0,709 | 0,889 |
| Open Images desenvolvimento / lateral | 12 | 0,684 | 0,347 | 0,741 | 0,538 | 1,000 |
| Open Images desenvolvimento / frontal/três quartos | 6 | 0,679 | 0,275 | 0,741 | 0,722 | 0,667 |
| Big Broiler teste anterior / global | 40 | 0,929 | 0,617 | 0,857 | 0,949 | 1,300 |
| Big Broiler teste anterior / oblíqua | 20 | 0,988 | 0,645 | 0,964 | 0,966 | 0,250 |
| Big Broiler teste anterior / superior | 20 | 0,788 | 0,551 | 0,722 | 0,926 | 2,350 |
| Open Images teste externo final / global | 30 | 0,721 | 0,374 | 0,730 | 0,744 | 0,667 |
| Open Images teste externo final / lateral | 24 | 0,618 | 0,299 | 0,570 | 0,742 | 0,708 |
| Open Images teste externo final / frontal/três quartos | 6 | 0,826 | 0,588 | 0,823 | 0,833 | 0,500 |

As métricas completas estão em [diverse_metrics.json](diverse_metrics.json) e [open_images_final_metrics.json](open_images_final_metrics.json). mAP, precisão e recall são do avaliador Ultralytics; a contagem usa confiança 0,25 e NMS IoU 0,5. Reduzir NMS de 0,7 para 0,5 foi decidido **na validação**: MAE Big Broiler caiu de 1,225 para 0,975, com 57 falsos positivos e 32 perdas entre as 356 caixas anotadas (correspondência gulosa, IoU ≥ 0,5). Nas fotos externas de desenvolvimento o MAE caiu de 1,500 para 0,889. O teste Big Broiler já havia sido aberto no baseline; é comparação posterior, não um teste totalmente independente. O Open Images final não orientou treino, seleção do peso ou limiar. O recorte externo é pequeno, especialmente o frontal, e não prova desempenho em câmera na altura das aves no interior de aviário.

## Inspeção de `input/` e teste de recortes

Depois de fixar peso e NMS pelos dados anotados, reprocessei as quatro fotos de `input/`. `cama-do-aviario-cursos-cpt.jpg` passou de 21 caixas espúrias para 6: não há mais caixas sobre letras/logotipo ou frestas vazias, mas uma ave recebe duas caixas. `images (1).jpg` passou de 19 para 8 caixas, com menos duplicatas e algumas aves perdidas. `images.jpg` passou de 15 para 13 caixas, ainda com perdas e caixas abrangendo grupos. A panorâmica `.webp` passou de 27 para apenas 1 detecção: há muitas aves próximas visíveis sem caixa, além das minúsculas ao fundo que podem ser desconsideradas na inspeção. Portanto, a boa avaliação nos conjuntos anotados **não** se transfere automaticamente à cena escura, profunda e muito densa.

Testei inferência em quatro recortes sobrepostos e NMS global 0,5, sem incorporar ao comando. Na validação anotada Big Broiler, o MAE aumentou de 0,975 para cerca de 3,95 aves/imagem; por vista, de 0,650 para 2,550 na oblíqua e de 1,300 para 5,350 na superior. Na panorâmica, a inspeção exploratória passou de 1 para 10 caixas, ainda deixando a maioria das aves próximas sem detecção e incluindo caixa sobre comedouro. O teste por recortes foi rejeitado. A próxima investigação deve usar cenas densas **com caixas completas**, sem tomar a imagem de `input/` como gabarito.

## Rodada 4 — cenas densas anotadas

Para tratar essa lacuna, obtive o [PIO](https://zenodo.org/records/16686320), licença CC BY 4.0, arquivo `data.rar` com MD5 `c207661257123da12df2ea700fe4c2d1`. O conjunto contém câmeras superiores em instalação comercial (`C`) e protótipo (`P`), com caixas YOLO individuais de galinhas; ele complementa, sem substituir, as vistas oblíqua e lateral do treino. A inspeção visual de três exemplos confirmou correspondência das caixas às aves. No arquivo foram encontradas duas caixas degeneradas na amostra selecionada; ambas foram descartadas, sem alterar as originais.

Separei por instalação e semana: treino `C-W3`, `C-W4`, `P-W3`, `P-W4` (20 imagens espaçadas de cada, 80 no total e 13.068 caixas após limpeza); validação `C-W5`, `P-W5` (10 de cada, 20 imagens e 1.508 caixas); teste fechado `C-W6`, `P-W6` (10 de cada, 20 imagens e 1.566 caixas). Os quadros de uma semana não atravessam divisões. A câmera e o local permanecem os mesmos entre semanas, portanto isso avalia mudança temporal e de crescimento, **não** transferência para outro aviário. O treino combinado tem ainda as 400 imagens Big Broiler e 159 Open Images de antes; a dependência não é exclusivamente da vista superior. O teste PIO ficou fechado antes do ajuste.

Antes de treinar nessa amostra, o peso da rodada 3 obteve na validação PIO mAP50 0,639, mAP50:95 0,326, precisão 0,743, recall 0,548 e MAE de contagem 43,5 aves/imagem (média anotada 75,4). A meta de reduzir o MAE pela metade e manter o desempenho das outras vistas foi fixada no GOALS.md local antes do novo treino. Por ser uma cena superior, mesmo uma melhora no PIO não prova funcionamento na panorâmica oblíqua de `input/`.

Treinei o peso diverso da rodada 3 por mais 5 épocas com as 80 imagens PIO de treino adicionadas, tamanho 512, batch 4, semente 42. O comando foi `python src/train.py --manifest data/manifest_pio.csv --model models/galinhas_diverse.pt --epochs 5 --imgsz 512 --batch 4 --seed 42 --iou 0.5 --processed data/processed_pio --run-name pio_v1 --output-model models/galinhas.pt --output-metrics data/raw/pio_val_metrics.json`. A escolha foi feita usando apenas a validação Big Broiler/PIO e as 18 fotos externas de desenvolvimento. O novo peso atingiu a meta PIO de MAE até 21,75 (16,15) e mAP50 ao menos 0,639 (0,864), preservou MAE de até 3 nas duas vistas Big Broiler (2,45 oblíqua e 1,00 superior) e mAP50 lateral externo acima de 0,60 (0,706). Assim, o peso foi fixado antes de abrir o teste PIO e o segundo teste externo.

| Validação anotada | Imagens | mAP50 | mAP50:95 | Precisão | Recall | MAE contagem |
|---|---:|---:|---:|---:|---:|---:|
| Global Big Broiler + PIO | 60 | 0,847 | 0,516 | 0,840 | 0,778 | 6,533 |
| Big Broiler oblíqua | 20 | 0,940 | 0,590 | 0,890 | 0,910 | 2,450 |
| Big Broiler superior | 20 | 0,750 | 0,560 | 0,750 | 0,860 | 1,000 |
| PIO superior densa | 20 | 0,864 | 0,515 | 0,835 | 0,777 | 16,150 |

## Teste do peso escolhido - 11/11/2026

O teste PIO usa somente semana 6, separada das semanas 3/4 de treino e 5 de validação; contém 1.566 aves anotadas em 20 imagens, média de 78,3. O teste Big Broiler usa vídeos inteiros separados por divisão, mas já havia sido examinado no baseline e serve como comparação posterior, não como teste inédito. As métricas completas, incluindo erros por zona e exemplos de contagem por imagem, estão em [final_metrics.json](final_metrics.json). Confiança 0,25 e NMS IoU 0,5 foram fixados na validação para **contagem e inferência**; mAP, precisão e recall seguem a validação padrão do Ultralytics.

| Teste anotado | Imagens | mAP50 | mAP50:95 | Precisão | Recall | MAE contagem |
|---|---:|---:|---:|---:|---:|---:|
| Global Big Broiler + PIO | 60 | 0,865 | 0,529 | 0,850 | 0,814 | 3,467 |
| Big Broiler oblíqua | 20 | 0,989 | 0,67 | 0,96 | 0,96 | 0,600 |
| Big Broiler superior | 20 | 0,822 | 0,63 | 0,77 | 0,95 | 2,050 |
| PIO superior densa | 20 | 0,858 | 0,503 | 0,848 | 0,783 | 7,750 |

Exemplos do peso final com gabarito no teste: no quadro oblíquo `050_2_31_1920_011_1_05_000357.jpg`, 11 aves anotadas, 10 caixas previstas, 0 falso positivo e 1 ave perdida; no superior `087_3_32_1920_010_0_10_000714.jpg`, 8 anotadas, 9 previstas, 1 falso positivo e nenhuma perdida; no PIO `C-W6-V0004.jpg`, 65 anotadas, 68 previstas, 8 falsos positivos e 5 perdidas. A associação diagnóstica foi gulosa com IoU ≥ 0,5, confiança 0,25 e NMS 0,5; não é o cálculo do mAP. No PIO, perdas e falsos positivos podem se compensar na contagem, por isso o MAE isolado não basta.

Um segundo recorte independente de 30 fotos e 49 caixas da divisão oficial de teste do Open Images V5 foi fechado antes da execução do peso PIO. São 24 vistas laterais e 6 frontais/três quartos, cada foto de autor diferente e sem autores do treino ou desenvolvimento. A seleção com atribuições está em [open_images_final_round2.jsonl](../data/open_images_final_round2.jsonl), e as [métricas completas](open_images_final_round2_metrics.json) resultam de `python src/evaluate_open_images.py --selection data/open_images_final_round2.jsonl --model models/galinhas.pt --iou 0.5 --output docs/open_images_final_round2_metrics.json`.

| Teste externo | Imagens | mAP50 | mAP50:95 | Precisão | Recall | MAE contagem |
|---|---:|---:|---:|---:|---:|---:|
| Global | 30 | 0,748 | 0,428 | 0,753 | 0,633 | 0,500 |
| Lateral | 24 | 0,719 | 0,413 | 0,764 | 0,648 | 0,458 |
| Frontal/três quartos | 6 | 0,816 | 0,479 | 0,870 | 0,786 | 0,667 |

O mAP global não substitui essas linhas por vista. O recorte externo contém poucas aves por foto, algumas possivelmente não anotadas; o MAE se refere às caixas oficiais. A detecção superior densa é satisfatória frente à meta pré-fixada, mas o recall de 0,783 ainda implica aves perdidas, e a média de 7,75 aves de erro por imagem não serve para contagem precisa de manejo. A regra por zonas concorda ou discorda da mesma regra aplicada às caixas, mas não diagnostica aglomeração biologicamente relevante. Reavaliei os 12 rótulos visuais exploratórios com o **peso final**, confiança 0,25 e NMS IoU 0,5: 9/12 acertos, 6/6 oblíquos e 3/6 superiores. Os três erros superiores são falsos positivos de aglomeração em `_000714`, `_011429` e `_014286`, nos quais o rótulo visual era 0 e a regra previu 1. O [JSON atualizado](agglomeration_eval.json) registra esses casos. O tamanho e a rotulagem por IA não validam uso zootécnico.

## Verificação final de `input/` e vídeo

Somente após a escolha por gabaritos, executei `python src/infer.py` nas quatro fotos de `input/`, substituindo seus resultados correspondentes. A inspeção visual foi feita nas imagens geradas, sem tratar os totais como acurácia, pois não há caixas exaustivas dessas fotos. Aves minúsculas ao fundo da panorâmica foram desconsideradas nesta análise qualitativa.

| Entrada | Detecções | Erros visíveis e interpretação |
|---|---:|---|
| `images.jpg` | 9 | várias aves médias ao fundo sem caixa, algumas caixas cobrem aves sobrepostas e uma caixa parece recair no alimentador; a perspectiva e a iluminação diferem do Big Broiler. |
| `images (1).jpg` | 5 | caixas em aves próximas, mas outras aves igualmente visíveis ficam sem caixa; há oclusão entre corpos brancos. |
| `cama-do-aviario-cursos-cpt.jpg` | 5 | quatro caixas em aves e uma no bebedouro/comedor central; nenhuma caixa sobre letras/logotipo ou frestas vazias, ao contrário do baseline. |
| `0rF2zO6V-frango-corte-toledo-ari-dias-jpeg.webp` | 4 | três caixas em aves próximas e uma caixa enorme que engloba parte da cena; muitas aves próximas e médias ficam sem caixa, apesar de as distantes minúsculas poderem ser ignoradas. A contagem e `aglomeração_visual=0` não descrevem adequadamente essa imagem. |

As causas prováveis, inferidas visualmente e ainda não isoladas em experimento, são domínio diferente do treino (aviário panorâmico escuro, profundidade acentuada, resolução relativa menor por ave), oclusão forte e objetos claros parecidos com aves. O falso positivo grande evidencia que caixas individuais aprendidas nas fontes não se transferem bem à cena panorâmica. Não ajustei limiar ou peso com base nessas fotos sem gabarito. Para reivindicar melhora nessa perspectiva é necessário um conjunto de capturas oblíquas densas com caixas completas, separado por sessão/local para treino, validação e teste.

Também criei um clipe real de cinco quadros a partir de vídeo Big Broiler de validação, em pasta local ignorada, e executei a inferência duas vezes. A saída MP4 foi lida com cinco quadros; o CSV tem cinco linhas de contagem. Na segunda execução, os mesmos dois nomes foram atualizados e o hash do vídeo de entrada permaneceu igual. As imagens e os vídeos de `input/`/`output/`, assim como o clipe de verificação, não entram no Git.
