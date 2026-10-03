# Experiments — ideias e anotações

Caderno da pasta `experiments/`. Nada aqui roda sozinho; é lista de coisas pra
experimentar e coisas aprendidas. Ordenado do mais barato ao mais caro.

> **Reorganização de 18/09.** A pasta se chamava `playground/` e guardava tanto
> script quanto biblioteca. Agora `experiments/` só tem coisa que *roda*, e nada
> aqui pode ser importado por ninguém. O que era importado foi pra `structsept/`:
> `plate_with_hole.py`, `plate_hole_params.py`, `pointcloud_sdf.py` e o novo
> `fem.py` (malha de tetraedros + montagem de `K`, extraído do
> `plate_with_hole_stiffness.py`). A regra completa está em `docs/structure.md`.

## O que já existe aqui

- `plate_geometry.py` — placa em lattice sobre uma caixa física, só geometria.
  O mais simples de todos: nenhum furo, nenhum FEM.
- `plate_with_hole_network.py` — placa com a rede treinada de verdade
  (`RoundCross`) no lugar do decoder analítico, e só geometria: monta o domínio
  analítico Ω (placa + furo), monta o `f_θ`, junta os dois e gera três figuras —
  os três campos SDF lado a lado, a varredura do latente e o render da malha.
  É o item 7 da fila, feito.
- `plate_with_hole_stiffness.py` — pega a mesma placa, malha em tetraedros e
  monta a matriz de rigidez global `K` com `torch-fem`. Para *antes* de condições
  de contorno, carga e solve. Salva `K` em `.npz`, a malha em `.vtk` e o padrão
  de esparsidade em `.png`. É o item 9 da fila, feito até a metade. Depois da
  reorganização virou um *driver* de 175 linhas: a conta toda está em
  `structsept/fem.py`.
- `pointcloud_to_lattice.py` — nuvem de pontos entra, campo latente sai.

A geometria em si não mora mais aqui: é `structsept.plate_with_hole`, com a
placa 1×1×0,1 m, raio e centro (x, y) do furo em metros, e a flag `--solid` pra
versão maciça sem rede neural nenhuma. Roda com
`uv run python -m structsept.plate_with_hole`.

## Fila de experimentos

### Baratos (segundos, sem FEM)

1. **Varrer o raio do furo.** Rodar `plate_with_hole` com raio 0,1 / 0,2 / 0,3 e
   comparar o volume sólido que o script reporta. Dá pra checar contra a conta
   analítica `L·W·t − πr²t` no caso `--solid`.
2. **Furo saindo pela borda.** Centro em (0.9, 0.5) com raio 0,2 — metade do
   furo cai fora da placa. Ver se a malha continua estanque. É um bom teste do
   `CappedBorderSDF`.
3. **Mais de um furo.** `DifferenceSDF` aceita vários objetos de uma vez:
   `DifferenceSDF(base, furo1, furo2, ...)`. Só mudar a chamada.
4. **Trocar o furo por outra forma.** `sdf_primitives` tem `BoxSDF`,
   `TorusSDF`, `HexagonSDF`, `PolygonSDF`... Furo hexagonal é uma linha.
5. **`SmoothDifferenceSDF` em vez de `DifferenceSDF`.** Dá fillet automático na
   borda do furo, controlado pelo parâmetro `k`. Ver o efeito num corte.

### Médios

6. **Demo do sphere tracing** (`why_sdf.py`, hoje no scratchpad temporário).
   Mostra em números por que a *distância* vale mais que "dentro/fora":
   10 avaliações contra 18.038 pra achar onde um raio bate. Vale trazer pra cá.
7. ~~**Rede neural de verdade.**~~ Feito em `plate_with_hole_network.py`. Só que
   o latente *não* significa a mesma coisa nos dois — ver anotação abaixo — então
   o erro de reconstrução ponto a ponto ainda está em aberto: precisa antes achar
   o λ da rede que corresponde a cada raio analítico.
8. **Lattice graduado.** Trocar `Constant([0.4])` por `SplineParametrization`,
   com pontos de controle diferentes: barras grossas de um lado, finas do outro.
   É *a* mudança que abre o espaço de projeto do paper. Continua sem FEM.

### Caros (aí entra o FEM)

9. Malha tetraédrica (`--volume`) + `torch-fem`. Ver
   `DeepSDFStruct/tests/test_structural_optimization.py`, que é um exemplo
   completo e rodável.

## Entrada por point cloud (13/09, recado do supervisor)

O supervisor disse que o código deve **receber uma nuvem de pontos** e a rede
deduzir a geometria a partir dela. Levantamento do que já existe:

- **Isso já está implementado na biblioteca**, para *malha* alvo:
  `DeepSDFStruct/geom_reconstruction.py` →
  `LocalShapesReconstructor.fit_mesh(mesh, tiling=[...])` normaliza a malha,
  monta o `LatticeSDFStruct`, amostra o SDF de referência e otimiza os pontos de
  controle do spline de latentes. `fit_samples(struct, samples)` é a versão que
  recebe as amostras já prontas. Ambos chamam
  `deep_sdf/reconstruction.py::reconstruct_from_samples`. Exemplo rodável:
  `DeepSDFStruct/tests/test_reconstruction.py` (cone.stl, tiling 2×2×2).
- **DeepSDF não tem encoder.** Não existe forward nuvem → latente. É
  *auto-decoder*: decoder congelado, o latente (aqui: os pontos de controle) é
  achado por Adam/LBFGS contra as amostras. Resultado prático é o mesmo, mas o
  custo é uma otimização de milhares de passos por peça, não um forward. Se o
  supervisor espera um encoder de verdade (PointNet → latente), isso é trabalho
  novo e treino novo.
- **O que falta pra aceitar nuvem crua:** `reconstruct_from_samples` consome
  `SampledSDF(samples (N,3), distances (N,1))` — precisa de distância *com
  sinal*. Hoje o sinal vem de `SDFfromMesh` (igl + winding number), que exige
  malha estanque. Uma nuvem só com x,y,z não tem isso. Caminhos:
  1. nuvem amostrada de uma malha/CAD → amostrar direto, zero trabalho novo;
  2. nuvem com normais → Poisson (open3d, não instalado) → malha → caminho atual;
  3. nuvem crua → perda só de superfície (φ = 0 nos pontos) + eikonal +
     pontos livres penalizados, estilo IGR/SIREN. `reconstruct_from_samples` **já
     tem `eikonal_lambda`**, então a peça principal existe; faltaria montar o
     `SampledSDF` com `distances=0` e acrescentar os pontos fora da superfície.
- Modelos treinados estão locais (`DeepSDFStruct/trained_models/`) e há
  `tests/data/*.stl` (cone, stanford_bunny, flow_channel) pra testar offline.

**Respondido (13/09):** a nuvem é a **peça inteira** e o lattice deve **imitar** a
nuvem. Ou seja, ajustar o campo de latentes contra a nuvem — não só recortar o
lattice pelo contorno dela.

### Feito

- `structsept/pointcloud_sdf.py` — a peça que faltava: `PointCloudSDF`, um `SDFBase`
  que tira o SDF direto da nuvem, sem malha. Sinal pelo **winding number
  generalizado** (soma de dipolos, Barill et al. 2018); magnitude por KD-tree do
  `scipy`, refinada pro plano tangente perto da superfície. Sem dependência nova
  (`open3d` continua fora). Traz também `cloud_from_mesh` (simula um scan a
  partir de um STL), `estimate_normals` (PCA + orientação pelo centroide, só
  serve pra peça estrelada) e `sample_cloud_sdf`.
- `experiments/pointcloud_to_lattice.py` — ponta a ponta: nuvem → normalização →
  `LatticeSDFStruct` → amostras → `fit_samples` → malha + `.npz` com os pontos de
  controle (que *são* as variáveis de projeto do MMA).

Números do run padrão (cone.stl, 20 000 pontos, tiling 3×3×3, 5 épocas). Tempo
total 71 s com `--resolution 16`, 146 s com os 32 do default — a diferença é toda
no export da malha, não no ajuste:

| etapa | métrica |
|---|---|
| nuvem → SDF (vs. malha original) | IoU 100,0 %, sinal 100 %, erro na faixa 0,0012 |
| ajuste do campo de latentes | IoU 15,4 % → **99,5 %**, 170 passos em 18 s |

Truques que valem lembrar:

- **A banda perto da superfície é construída, não consultada.** Um ponto da nuvem
  deslocado de `t` ao longo da normal tem SDF `= t`. Isso evita o winding number
  (que é O(n_query × n_cloud)) justamente onde estão 85 % das amostras; a soma de
  dipolos só roda nos pontos uniformes. Vale enquanto `t` for menor que a menor
  espessura da peça.
- **`build_struct` aceita um `trimesh.PointCloud`.** Ele só usa `.copy()`,
  `.bounds` e `.apply_transform`, que `PointCloud` tem. Não precisa fabricar
  malha falsa. As normais sobrevivem à normalização (escala uniforme + translação).
- **Não comparar φ ponto a ponto longe da superfície.** Dentro de um lattice o φ é
  a distância à barra mais próxima, não ao contorno da peça: contra um sólido isso
  dá erro grande mesmo com ajuste perfeito. A métrica honesta é IoU de ocupação,
  mais o erro restrito à faixa `|φ_alvo| < 0.05`.
- **`plot_reconstruction_loss` quebra com menos de ~60 passos** (suaviza com janela
  de 41): `ValueError: Must have equal len keys and value`. Candidato a reportar
  upstream.

### Em aberto

- Se o alvo for uma peça **sólida**, "imitar a nuvem" faz o ajuste empurrar as
  células pro mais maciço que o decoder consegue — que provavelmente não é o que
  se quer. Confirmar com o supervisor: a peça alvo já é porosa/lattice, ou é
  sólida e o lattice deveria só *caber dentro* dela (aí é recorte, não ajuste,
  e os latentes ficam livres pro MMA)?
- Nuvem real sem normais: hoje só PCA + centroide. Orientação correta (propagação
  por árvore geradora mínima) não está implementada.

## Anotações

- **`AnalyticRoundCross` não é rede neural.** Apesar do nome "decoder" e dos
  83.578 parâmetros guardados, o `forward` ignora todos e calcula uma fórmula
  analítica: a união (`min`) de três cilindros de raio `r`, onde `r` é o próprio
  vetor latente. O termo inicial `‖x‖∞` do `min` é só o valor de partida e não
  adiciona material. Verificado destruindo os pesos com ruído: a saída não muda.
  Não é bug — é a geometria de referência do test case 1 do paper, com erro de
  reconstrução zero. Redes de verdade: `RoundCross`, `ChiAndCross`,
  `Primitives*`.

- **Faixa do latente: [0.15, 0.75].** Fora disso a rede não foi treinada.

- **O λ da `RoundCross` treinada não é o raio.** No decoder analítico o latente
  *é* o raio da barra, literalmente (`r = input[:, 0]` no `forward`). Na rede
  treinada os códigos latentes foram aprendidos, não prescritos, então a escala é
  outra: no mesmo λ = 0,6 a célula analítica ocupa 54 % do cubo unitário e a
  treinada 29 % (fração de amostras com φ < 0, 60 000 pontos em [−1,1]³). Os dois
  arquivos `LatentCodes/latest.pth` são idênticos e trazem valores negativos
  (−0,19), que como raio não querem dizer nada. Conclusão prática: comparar
  *formas*, nunca os números do λ. Comparação direta φ_rede − φ_analítico no
  mesmo λ dá 0,12 de erro médio e só 83 % de concordância de sinal — isso é a
  escala diferente, não erro de treino.

- **`LatticeSDFStruct` deixa lixo no microtile.** Ele escreve um latente *por
  ponto de consulta* dentro do microtile (`microtile.latvec`), e isso fica
  gravado. Plotar o mesmo objeto depois quebra com "Latent vector shape
  mismatch: torch.Size([90000, 1]) does not align with 62500 queries". Solução
  barata: embrulhar o decoder de novo (`SDFfromDeepSDF(model)`) — os pesos são
  compartilhados, só o wrapper é novo.

- **`np.linspace` estraga o latente.** O escalar sai float64, vira tensor
  float64, e o decoder é float32: `mat1 and mat2 must have the same dtype`. Um
  `float()` explícito resolve.

- **Duas coordenadas diferentes convivem no código.** O SDF vive no cubo
  paramétrico [0,1]³; a `TorchSpline` só converte pra metros no fim, aplicada
  aos *vértices da malha*. Por isso a classe `ScaledSpaceSDF` existe: pra poder
  descrever o furo em metros e mesmo assim subtraí-lo do lattice.

- **Numa placa fina, a face manda.** Placa de 0,1 m de espessura: qualquer ponto
  no meio da chapa está a só 0,05 m das faces de cima e de baixo. Então pra
  quase todo ponto o φ é governado pela *espessura*, não pelo furo nem pela
  borda. Fácil de esquecer quando se pensa no desenho 2D.

- **`create_3D_mesh(mesh_type="volume")` perde ~30 % do volume.** Esse caminho
  usa a tetraedrização interna do FlexiCubes, e ela deixa cavidades: num cubo
  unitário maciço a soma dos tetraedros dá 0,68–0,70 em vez de 1,0, em qualquer
  resolução, e o contorno da malha tem 13 034 triângulos em 5 836 componentes
  (a superfície tem 1 200 em 1). O mesmo vale pra placa com furo. A malha de
  superfície, por outro lado, converge pro volume exato. O caminho correto é
  superfície → `mesh.tetrahedralize_surface` (tetgen), que reproduz o volume da
  superfície em todos os dígitos e ainda gera menos elementos. É o que
  `plate_with_hole_stiffness.py` faz. Atenção: o teste
  `test_structural_optimization.py` da biblioteca usa o caminho com cavidades.
  Vale reportar upstream.

- **`min`/`max` subestimam a distância.** Booleanas via `min`/`max` dão um valor
  menor que a distância real perto de cantos côncavos. É seguro (nunca
  superestima), mas é a primeira coisa a olhar se um sphere tracing atravessar
  a superfície.

## Dados de treino do DeepSDF — como se monta (16/09)

Levantamento feito antes de decidir se vale treinar rede própria.

- **Formato em disco:** um `.npz` por geometria, com os arrays `pos` e `neg`,
  cada linha `[x, y, z, φ]` em float32. Ou seja: é point cloud *com o SDF
  anexado*, não só coordenadas. O treino espera o layout
  `<data_source>/SdfSamples/<dataset>/<classe>/<instância>.npz` mais
  `<data_source>/splits/<nome>.json` listando as instâncias
  (`deep_sdf/workspace.py`, `deep_sdf/data.py::SDFSamples`).
- **Quem gera:** `sampling.py::SDFSampler`. `add_class()` aceita
  `trimesh.Trimesh`, spline do splinepy (vira triângulos via
  `extract.faces(n_faces)`), `torchSurfMesh` ou um `SDFBase` analítico;
  `process_geometries()` normaliza pra [-1,1]³, amostra e grava. `write_json()`
  escreve o split.
- **O φ é calculado uma vez, offline, pelo próprio repo** — `SDFfromMesh` (igl:
  closest point + winding number, exige malha estanque) ou o SDF analítico se a
  geometria for um `SDFBase`. O treino nunca recalcula: `SDFSamples` só lê npz e
  sorteia `SamplesPerScene` linhas por passo, metade de `pos` e metade de `neg`.
  Se a célula tem fórmula, passe o `SDFBase` — φ exato, sem erro de facetagem
  (é o que `generate_primitive_dataset.py` faz).
- **Amostragem:** uniforme em (-1,1)³ (`random_sample_sdf`) mais banda de
  superfície opcional (`sample_mesh_surface`: pontos deslocados ao longo da
  normal por `stds`, default [0.05, 0.025]). O paper usa uniforme puro, 500 k
  pontos por peça; `generate_primitive_dataset.py` usa 100 k uniformes + 1 M na
  banda com stds [0.005, 0.0001].
- **Inventário dos modelos prontos** (`DeepSDFStruct/trained_models/`):

  | modelo | arquitetura | dim. latente | nº de latentes |
  |---|---|---|---|
  | `AnalyticRoundCross` | analítica, sem rede | 1 (= raio) | 20 |
  | `RoundCross` | MLP 6×128 | 1 | 20 |
  | `ChiAndCross` | MLP 6×128 | 2 | 120 |
  | `PrimitivesCL08/16/32` | MLP 4×128, estilo DeepLS | 8 / 16 / 32 | 100 cenas |
  | `Primitives2D` | MLP 6×128, tanh | 16 | 19 079 |

  `RoundCross` e `ChiAndCross` cobrem só a família de célula treinada (test
  cases 1 e 2 do paper). Os `Primitives*` foram treinados em cenas aleatórias de
  esferas/caixas/cilindros e são os de propósito geral — é o que o
  `LocalShapesReconstructor` usa. Nos `Primitives*` os latentes salvos são
  zeros; o que vale é o `latent_fields_state_dict`.

## Hiperparâmetros do treino — o que o trainer lê de verdade (23/09)

Levantamento feito pra janela "All hyperparameters..." da aba Train
(`structsept/app/hyperparams.py` descreve cada chave do `specs.json`, com
faixa, default e o que faz). Lendo `deep_sdf/training.py` e o
`DeepSDFDecoder` linha por linha apareceram umas pegadinhas:

- **O gradient clipping é fixo em 1,0.** O trainer lê `GradientClipNorm` e
  logo depois, dentro do laço, sobrescreve com `grad_clip = 1.0`. Qualquer
  valor no specs é ignorado. Por isso a janela não oferece esse campo e lista
  ele em "Not editable here". Bug da biblioteca — vale reportar upstream.
- **`clampedL1` tem clamp próprio de 0,1.** `get_loss_function` constrói
  `ClampedL1Loss()` com o default `clamp_val=0.1`, independente de
  `ClampingDistance`. Com δ = 1,0 (como no `Primitives2D`) a faixa aprendida
  continua sendo ±0,1. Pra aprender a faixa larga de verdade: `L1`.
- **Com os defaults o learning rate nunca cai.** O `Step` divide por 2 a cada
  500 épocas, e os runs do app têm 30–200. A janela avisa isso como nota.
- **`SamplesPerScene` ímpar derruba o treino.** O loader tira `n // 2` de dentro
  e `n // 2` de fora (total `n − 1`), mas os índices dos latentes são repetidos
  `n` vezes: o `torch.cat` falha por uma linha. Confirmado num run de teste.
- **`latent_in = n_layers + 1` não é ignorado.** Índices de skip além das camadas
  somem em silêncio, *menos* esse: o construtor testa `layer + 1 in latent_in`
  até a camada de saída e a deixa com `1 − (d + 3)` neurônios → erro na
  construção. Índice 0 também quebra (dobra a entrada da primeira camada).
- **`CodeRegularization` (bool) não é lido.** Só o `CodeRegularizationLambda`
  conta; λ = 0 desliga. E o termo entra em rampa: `λ · min(1, época/100)`.
- **Os dois schedules são dois grupos do Adam**: o primeiro vale pros pesos do
  decoder e o segundo pros códigos latentes, nessa ordem.
- **O `seed` do specs não fixava os pesos iniciais.** `train_deep_sdf` constrói
  o decoder (`init_decoder`, pesos tirados do RNG global do torch) *antes* de
  chamar `torch.manual_seed`. No app, que vive muito tempo, o RNG já foi mexido
  por runs anteriores e pela aba Explore: dois runs idênticos saíam com pesos
  diferentes (|Δw| máx ≈ 0,9). Agora `structsept.app.training.train` semeia antes
  de chamar a biblioteca, e o teste `test_same_seed_gives_the_same_network`
  cobre isso. Qualquer script que chame `train_deep_sdf` direto tem o mesmo
  problema.
- **A última batch incompleta é descartada.** O `DataLoader` do trainer usa
  `drop_last=True` sempre que a batch cabe no dataset: com 15 formas e 10 por
  batch, 5 formas (sorteadas) ficam de fora de cada época. Um tamanho de batch
  que divida o número de formas usa todas.

Experimentos que isso abre (baratos, poucos minutos de CPU cada):

- Varrer o learning rate do decoder (1e-4 … 3e-3) num dataset pequeno com
  `Step` a cada ~50 épocas, e ver se o loss final de 200 épocas melhora em
  relação ao default que nunca decai.
- `clampedL1` com δ = 0,1 contra `L1` com δ = 0,3: o que muda na fração de
  volume e na espessura das barras quando a faixa aprendida é maior?

## Placa com furo num latente 2-D (23/09)

O experimento: o furo varia em três parâmetros `(x_c, y_c, r)` e a rede aprende
um latente de dimensão 2, `(λ₁, λ₂)`. Os parâmetros só geram o SDF — a rede
nunca os vê. Três parâmetros independentes não cabem sem perda em duas
dimensões; a pergunta é **qual deles o latente sacrifica**.

- **Dados:** `uv run python -m datagen.make_plate_hole --dim 2 --plot` escreve
  134 placas (Sobol 128 + 6 extremos, margem 0,05) em `data/`, com SDF exato,
  20 mil amostras por placa, em ~3 s e 32 MB. O `params.csv` ao lado guarda
  `(x_c, y_c, r)` de cada forma; a linha `i` é o código latente `i`.
  `datagen/` fica fora do `structsept/` de propósito: um escreve `data/`, o
  outro lê, e nenhum importa o outro.
- **Por que 2-D e não 3-D:** placa de espessura constante com furo passante →
  o campo 3-D é só a extrusão do 2-D. Em 3-D com t = 0,1 o plano médio fica
  quase constante (≈ −h) dentro do material, e a maior parte das amostras só
  ensina as duas faces, iguais em todas as formas. O 2-D vira o 3-D exato por
  extrusão quando precisar (`--dim 3` faz isso).
- **A biblioteca já treinava 2-D** (`geom_dimension` nos specs); quem travava em
  3-D era a GUI. Agora a dimensão vem do dataset (`dataset.json` ou largura das
  linhas do `.npz`), e decoders planares abrem na aba nova **Explore 2-D**.
- **Pegadinha:** `geom_dimension` errado nos specs derruba o treino no primeiro
  batch com `IndexError` (o `remove_nans` lê a coluna 3). A auditoria antiga do
  SDF maker fazia `reshape(-1, 4)` e lia arquivos 2-D como 3-D embaralhado
  quando o número de elementos dividia por 4 — sem acusar nada.
- **Orçamento de 30 min de CPU:** um passo custa ~35 µs por ponto de amostra
  (medido duas vezes). 134 formas × 2000 amostras ≈ 9,5 s por época, então
  ~150 épocas em 25 min. Sugestão: d = 2, 150 épocas, `SamplesPerScene` 2000,
  `Step` a cada 60 épocas com fator 0,5, sem dropout; o resto no default. O
  default (200 épocas × 8000 amostras) levaria ~2 h.
- **Como ver o resultado:** aba Explore 2-D, igual à Explore: um slider por
  componente de λ (λ₁, λ₂) e a forma `f_θ(λ, x, y)` mudando junto. Controle
  que vale a pena: o mesmo dataset com d = 3, que custa o mesmo por época.

## Treino de 30 min com a planilha "Quick setup" (23/09)

Planilha `NN_Training_Hyperparameters_Quick_Clean.xlsx`: 25 formas, 50 mil
amostras cada, 4096 pontos por passo, 5 formas por batch, d = 2, rede 4 × 64
ReLU sem dropout, 250 épocas, Adam 5e-4 / 1e-3, clamped L1 com δ = 0,1.

- **Como está, a planilha treina em 1 min, não em 30.** Medido pelo mesmo
  caminho da GUI (`structsept.app.training.train`): 58 s, 0,19 s por época. A
  rede 4 × 64 é ~6× mais barata que a 6 × 128 do default, então a conta do
  item anterior (35 µs por ponto) não vale pra ela: aqui dá ~2 µs por ponto.
- **Pra encher 30 min, mudei só duas coisas:** 25 → 134 formas (o paper usou
  120 pra d = 2; com 25 o R² do x_c saiu −0,26) e 250 → 1500 épocas. Com 1500
  o `Step` de 500 finalmente dispara (lr cai pela metade em 500 e 1000); com
  250 ele nunca caía. Medido: 1,11 s por época + 11 s de setup → ~28 min
  *sem GUI*. Na GUI deve dar ~32–35 min: o gráfico de loss da aba Train
  recarrega o `Logs.pth` e redesenha todas as losses por batch a cada 1,5 s, e
  isso fica mais caro conforme o run cresce (×1,1 no começo, ×1,35 perto da
  época 1400, medido num microbenchmark). Pra caber em 30 min na GUI: 1250
  épocas.
- **Traduções da planilha pro trainer:** "4096 pontos por passo" virou
  `SamplesPerScene` 4096 (por forma, convenção do DeepSDF → 20 480 pontos por
  passo). "Variância 0,01" virou `CodeInitStdDev` = √(0,01 · 2) ≈ 0,141, porque o
  trainer sorteia N(0, (σ/√d)²). "Amostragem uniforme": o trainer pega uma
  janela aleatória do array já embaralhado, metade dentro e metade fora; o
  dataset guarda metade uniforme + metade perto das bordas, porque com clamp de
  0,1 ponto longe da superfície quase não ensina nada.
- **Arquivos:** `data/SdfSamples/plate_hole_2d_n25` (Sobol 19 + 6 extremos) e
  `plate_hole_2d_n134` (Sobol 128 + 6), 25 mil uniformes + 25 mil na faixa por
  placa. Presets em `runs/preset_plate2d_sheet_quick` e
  `runs/preset_plate2d_30min` (só `specs.json`, não treinados; o dataset certo
  aparece na coluna dataset da tabela de runs). Na GUI: escolher o dataset →
  "All hyperparameters…" → Start from `run: preset_plate2d_30min` → Load →
  Apply → Train. Não apertar o botão "Quick preset" do card depois (ele põe 30
  épocas), e limpar o "Run name" antes de um segundo run na mesma sessão (senão
  ele oferece sobrescrever o anterior).
- **O que a planilha não diz e o trainer faz:** a rede "4 × 64" sai 64-60-64-64,
  porque o skip na camada 2 (default mantido) reinjeta (λ, x) e a camada 1
  encolhe 4 neurônios. A regularização do latente não começa em 1e-4: sobe em
  rampa de 0 até 1e-4 nas primeiras 100 épocas (fixo no trainer).
- **Smoke run (planilha pura, 25 formas):** loss 0,057 → 0,016, erro de
  ajuste mediano 0,016, R² x_c −0,26 · y_c 0,90 · r 0,32. O latente de 250
  épocas guardou o y_c e largou o x_c — ver se isso muda com 134 formas.

## Dois datasets: furo centrado (só r) e furo livre com d = 3 (23/09)

Pedido do supervisor, mesmo dia: separar o experimento em duas famílias da
mesma placa, uma com um parâmetro e outra com três, e treinar a segunda com
latente de dimensão 3 em vez de 2.

| dataset | o que varia | formas | como foi tirado | tamanho |
|---|---|---|---|---|
| `plate_hole_2d_r_only` | só `r`, furo em (0,5, 0,5) | 40 | grade uniforme, r de 0,07 a 0,45 | 24 MB |
| `plate_hole_2d_xyr` | `x_c`, `y_c`, `r` | 134 | Sobol 128 + 6 extremos, seed 0 | 80 MB |

Os dois com margem 0,05, 25 mil uniformes + 25 mil na faixa por placa (a
convenção da planilha), 2-D. Comandos:

```
uv run python -m datagen.make_plate_hole --dim 2 --radius-only --n-uniform 25000 --n-band 25000 --name plate_hole_2d_r_only --plot
uv run python -m datagen.make_plate_hole --dim 2 --n-uniform 25000 --n-band 25000 --name plate_hole_2d_xyr --plot
```

- **`--radius-only` é novo** (`PlateHoleSpace.sample_radius`): fixa o centro
  (default: o meio da placa, `--centre X Y` pra outro ponto) e varre só o
  raio, de `r_min` até o maior furo que cabe ali. Pra um parâmetro só, grade
  uniforme é mais útil que Sobol, então o `--method` default vira `grid`
  nesse modo. Os nomes das instâncias, o `params.csv` e o `unit` são os
  mesmos da família de três parâmetros — só `u_x` e `u_y` ficam constantes.
- **O manifesto agora diz o que variou:** `parameters.varied` (`["r"]` ou
  `["x_c", "y_c", "r"]`, lido dos valores gravados, não da flag), mais
  `parameters.fixed_centre` e `parameter_sampling.family` (esses dois vêm da
  flag `--radius-only`).
- `plate_hole_2d_xyr` é **byte a byte igual** ao `plate_hole_2d_n134` (mesma
  seed, mesmos parâmetros); só o nome mudou pra dizer o que varia. O `n134`
  pode ser apagado quando os presets antigos que apontam pra ele não
  interessarem mais.
- **Presets** (só `specs.json`, não treinados), com os mesmos hiperparâmetros
  do `preset_plate2d_30min` — 4 × 64 ReLU sem dropout, 4096 amostras por
  forma, 5 formas por batch, 1500 épocas, lr caindo pela metade em 500 e
  1000, clamped L1 com δ = 0,1 — mudando só o dataset, o `d` e o σ inicial
  do latente (√(0,01·d), como antes):

  | preset | dataset | d | σ | tempo estimado |
  |---|---|---|---|---|
  | `runs/preset_plate2d_r_only_d1` | `plate_hole_2d_r_only` | 1 | 0,100 | ~8 min (40 formas → 8 batches/época) |
  | `runs/preset_plate2d_xyr_d3` | `plate_hole_2d_xyr` | 3 | 0,173 | ~28 min sem GUI, ~33 na GUI |

  Os dois saem de `uv run python experiments/make_plate_presets.py`
  (`runs/` é gitignored e o `specs.json` guarda caminhos absolutos, então em
  outra máquina é gerar os datasets e rodar esse script).
  Na GUI: escolher o dataset → "All hyperparameters…" → Start from
  `run: preset_…` → Load → Apply → Train (mesma receita do item anterior).
  No `xyr` continuam sobrando 4 formas por época (134 não divide por 5);
  no `r_only` 40/5 fecha certinho.
- **O que olhar depois do treino:** no `r_only` o código é um número por
  placa; contra a coluna `r` do `params.csv` ele deve sair monotônico, e a
  aba Explore 2-D com um slider só deve varrer o furo de 0,07 a 0,45 sem
  passar por nada estranho. No `xyr` com d = 3 a pergunta é se os três
  parâmetros cabem agora: como o latente aprendido é uma base arbitrária,
  comparar por regressão linear de `(x_c, y_c, r)` sobre `(λ₁, λ₂, λ₃)`
  (R² de cada parâmetro), não componente a componente. Ainda não há script
  pra essa comparação — é o próximo passo natural.

## Atalho no desktop (27/09)

O GUI agora abre por atalho: `install_shortcuts.bat` na raiz cria
`Desktop\NN\Lattice explorer` e `Desktop\NN\SDF maker`. O que aprendi fazendo:

- Atalho sem console = `pythonw.exe`, e aí `sys.stderr` é `None`. A barra do
  tqdm do trainer (`deep_sdf/training.py`) escreve nela sem perguntar, então
  treinar por atalho morria no primeiro epoch com `'NoneType' object has no
  attribute 'write'`. Testar num terminal **não reproduz** — precisa forçar os
  streams pra `None`. O launcher (`structsept/app/launcher.py`) manda tudo pra
  `outputs/logs/<janela>.log` e desliga o tqdm (`TQDM_DISABLE=1`); a aba Train
  lê o progresso do `Logs.pth` mesmo.
- Importar torch + DeepSDFStruct leva 15–30 s aqui; sem splash a pessoa clica
  duas vezes. O splash abre em menos de 1 s e a importação roda numa thread.
- Atualiza sozinho: código, por ser install editável, já roda direto da pasta.
  Dependência, o launcher compara o hash de `pyproject.toml` + `uv.lock` +
  `DeepSDFStruct/pyproject.toml` com o do último `uv sync` e sincroniza quando
  muda — menos com outra janela do structsept aberta (Windows não troca DLL
  carregada; o log diz que pulou). `.lnk` guarda caminho absoluto,
  então não vai pro git — cada máquina gera o seu com o `.bat`. Pro supervisor:
  instalar `uv`, `git clone --recursive`, clicar no `.bat`.

## Treino longo da placa só-raio: 8 × 256, até 4 h, sem ninguém olhando (27/09)

Pedido: treinar a família `plate_hole_2d_r_only` (furo centrado, só `r` varia,
40 raios) com hiperparâmetros "altos", começando 2 h depois do pedido e durando
no máximo 4 h. Virou o script `experiments/train_plate_r_only_4h.py`, que
espera até a hora marcada, treina e para sozinho no prazo.

- **Tamanho da rede escolhido pelo relógio, não pelo gosto.** Torch aqui é só
  CPU (`2.13.0+cpu`; a GTX 1660 Ti fica de fora sem trocar o pacote, e trocar
  dependência é decisão do Tomás). Benchmark de 3 épocas, 4096 amostras por
  forma, 40 formas, 6 threads:

  | decoder | µs por ponto | s por época |
  |---|---|---|
  | 6 × 128 | 38 | 6,2 |
  | **8 × 256** | **106** | **17,4** |
  | 8 × 512 | 220 | 36,0 |

  O 8 × 512 do DeepSDF original daria ~300 épocas em 3 h; o 8 × 256 dá 600
  (~2,9 h medido no frio, sobra ~1 h pro laptop esquentar e desacelerar).
- **Receita** (`runs/plate2d_r_only_d1_8x256_4h`): 8 × 256 ReLU, weight norm,
  skip na camada 4, dropout 0,2 em todas (default do DeepSDF, segura a rede
  grande), d = 1 com σ = 0,1, 4096 amostras por forma, **4 formas por batch**
  (40/4 = 10 passos por época, nenhuma forma sobra), clamped L1 δ = 0,1, Adam
  5e-4 / 1e-3 caindo pela metade a cada 150 épocas, 600 épocas, `latest.pth`
  a cada 10, snapshot a cada 100. Batch de 4 em vez de 5 foi pra ter mais
  passos de otimizador pelo mesmo custo por época (o custo é pontos por época,
  não passos).
- **Como ele para no prazo:** o trainer da biblioteca não tem limite de tempo,
  mas instala um handler de Ctrl-C que faz `sys.exit(0)`. Uma thread de
  vigia chama `_thread.interrupt_main()` no horário limite, o handler roda na
  thread principal e o loop sai limpo; o que fica é o último `latest.pth` (no
  máximo 10 épocas perdidas) mais os snapshots numerados. O `metadata.json`
  grava `stopped_at_deadline` e `last_epoch`. Testado nos dois caminhos com
  runs de 2 épocas e de 50 s antes de agendar o de verdade.
- **Não deixar o Windows dormir:** `SetThreadExecutionState(ES_SYSTEM_REQUIRED)`
  enquanto espera e treina — pedido por processo, some quando ele acaba, não
  mexe em configuração. Na tomada o laptop já não dormia (sleep = nunca no AC);
  na bateria dormiria em 30 min, por isso o pedido.
- **Onde olhar depois:** `runs/plate2d_r_only_d1_8x256_4h/code_vs_r.png` e
  `code_vs_r.csv` (código aprendido × raio, Pearson/Spearman e se é monotônico —
  o script faz essa comparação sozinho no fim), `Logs.png` (loss), e o log
  completo em `outputs/logs/plate2d_r_only_d1_8x256_4h.log`. O run aparece na
  tabela da GUI como qualquer outro; a aba Explore 2-D com um slider deve
  varrer o furo de 0,07 a 0,45. Pra matar um run em andamento: o PID está em
  `outputs/logs/<run>.pid`, `taskkill /PID <pid> /F`.
- **Pra repetir com outro horário / outra duração:**
  `uv run python experiments/train_plate_r_only_4h.py --start-at 01:10 --max-hours 4`
  (`--start-at now --epochs 5` pra um teste rápido). Lançado escondido com
  `Start-Process -WindowStyle Hidden` pra sobreviver ao fechamento do terminal.

**Resultado (28/09):** rodou de 01:10 a 02:46 — 1 h 36 min pras 600 épocas,
9,6 s por época (a máquina de madrugada, sem mais nada rodando, foi quase 2×
mais rápida que o benchmark de dia). Loss 0,053 → 0,0043, ainda caindo em 600
(0,0046 em 500): cabia 1000+ épocas no mesmo prazo de 4 h. O código latente
aprendido é uma reta contra o raio: Pearson −1,000, Spearman −1,000,
monotônico, de +0,216 (r = 0,07) a −0,173 (r = 0,45). Ou seja, com um
parâmetro gerador, d = 1 recupera o parâmetro. A planilha de hiperparâmetros
do supervisor está agora em `docs/hyperparameters/` (o template intacto +
uma cópia com só a coluna Value preenchida pra este run).

## GUI: runs por data, planilha do supervisor, de onde vieram os valores (28/09)

Três pedidos pequenos na aba Train, todos em `tests/test_app_hyperparams.py`:

- **Tabela de runs em ordem de data**, mais novo em cima. A data é o
  `timestamp` do `metadata.json` (escrito quando o run acaba, ou quando um
  preset é gerado). Run sem metadata (começado à mão, ou ainda treinando) é
  datado pelo `specs.json` e aparece com `~` na frente. Clicar num cabeçalho
  ordena por aquela coluna; clicar de novo inverte. A lista "Start from" da
  janela de hiperparâmetros segue a mesma ordem.
- **"Import sheet..." na janela de hiperparâmetros** lê a planilha do
  supervisor (`docs/hyperparameters/*.xlsx`). O template não fala a língua do
  trainer um-pra-um, então `hyperparams.from_sheet` faz a conta e diz na linha
  de status: *Points per training step* é o batch inteiro (amostras por forma
  = pontos ÷ *Geometries per batch*, arredondado pra baixo até par);
  *Latent initialization variance* é por componente e o trainer sorteia
  N(0, σ²/d), logo σ = √(variância · d); fator e intervalo de decaimento vão
  pros dois schedules como Step. Linhas que não são configuração (nº de
  geometrias, amostras por geometria, estratégia de amostragem, ativação,
  otimizador) são conferidas contra o que o app faz e reportadas quando
  discordam; linha em branco fica no default; linha que o template não tem é
  nomeada e ignorada. Conferido: a planilha preenchida do run de 4 h
  reproduz exatamente a receita do `train_plate_r_only_4h.py` (8 × 256,
  σ = 0,1, 4096/4, step 150, 600 épocas).
  O `.xlsx` é lido com `structsept/app/xlsx.py` — zip + XML da stdlib, só as
  células de uma planilha — em vez de adicionar `openpyxl` ao ambiente
  (adicionar dependência é decisão do Tomás; trocar por openpyxl é uma função).
- **A janela lembra de onde vieram os valores.** Load ou Import + Apply: o
  card diz "From run: X (loaded 14:02)", e a janela, aberta de novo, já vem
  com essa fonte selecionada em "Start from" e a mesma frase no status — mais
  "edited since: epochs, width" pro que foi mexido à mão depois, no card ou na
  janela. A comparação é contra os valores como foram carregados
  (`hyperparams.origin_summary`), então um conjunto que *começou* como o de um
  run nunca é apresentado como se fosse o do run. "Reset to defaults" esquece
  a fonte.

## Placa só-raio com d = 2: a mesma receita, um componente a mais (28/09)

Pedido: repetir o run de 4 h (`plate2d_r_only_d1_8x256_4h`) com **dois**
componentes latentes, sabendo que só o raio varia — a pergunta é o que a rede
faz com a dimensão que sobra.

- **Planilha:** `docs/hyperparameters/NN_Training_Hyperparameters_plate2d_r_only_d2_8x256_4h.xlsx`
  é cópia da do d = 1 com uma célula mudada (*Latent dimension* = 2). A
  variância inicial continua 0,01 por componente, então o σ do trainer vira
  √(0,01 · 2) = 0,141 — `hyperparams.from_sheet` faz a conta e o teste
  `test_the_d2_sheet_is_the_same_recipe_with_a_two_component_code` confere.
- **Script:** `experiments/train_plate_r_only_4h.py` ganhou `--latent-dim`
  (default 1; nada muda pro run antigo). O nome do run vira
  `plate2d_r_only_d<d>_8x256_4h`, o σ sai de `CODE_VARIANCE · d`, e a
  comparação do fim olha cada componente contra o raio (Pearson, Spearman,
  monotônico) e os códigos juntos: quanto da variância cai no primeiro eixo
  principal (100 % = os 40 códigos numa reta; um parâmetro gerador não precisa
  de mais que isso) e se a posição ao longo desse eixo segue o raio.
  `code_vs_r.csv` tem `code_0, code_1`; `code_vs_r.png` mostra os componentes
  contra r à esquerda e o plano latente colorido por r à direita. Testado com
  2 épocas nos dois caminhos (d = 1 e d = 2) antes de lançar.
- **Lançado 28/09 23:12**, escondido (`Start-Process -WindowStyle Hidden`),
  `--latent-dim 2 --start-at now --max-hours 4`. 600 épocas devem levar
  ~1 h 40 como no d = 1 (decoder do mesmo tamanho, só a entrada tem uma
  coluna a mais). Log em `outputs/logs/plate2d_r_only_d2_8x256_4h.log`, PID
  em `outputs/logs/plate2d_r_only_d2_8x256_4h.pid`.
- **O que esperar:** (a) os códigos numa reta e um componente carrega tudo —
  o outro fica parado perto do zero (λ = 1e-4 puxa pra lá); (b) reta
  inclinada — os dois componentes correlacionam com r, redundantes, PC1
  continua ~100 %; (c) curva ou nuvem — a rede usa o segundo eixo pra algo
  que não é o raio (ruído da inicialização congelado), PC1 bem abaixo de
  100 % e Spearman baixo num dos componentes. Só (c) seria surpresa.

**Resultado (29/09):** rodou de 23:12 a 01:09 — 1 h 57 min pras 600 épocas
(11,7 s por época; mais lento que o d = 1 de madrugada porque a GUI estava
aberta e os testes de fumaça rodaram junto no começo). Loss final 0,0045,
igual ao d = 1 (0,0043): o componente a mais não ajudou nem atrapalhou o
ajuste. Saiu o caso (b), com a perpendicular sendo ruído congelado:

- Os 40 códigos ficam numa **reta inclinada** a −50° no plano latente (eixo
  principal (0,64, −0,77)), com 84,8 % da variância; a posição ao longo dela
  é o raio: Pearson +0,999, Spearman +1,000, monotônica.
- Cada componente sozinho correlaciona com r (código 0: +0,90, código 1:
  −0,93) mas **não é monotônico** (22 trocas de sinal cada): é a projeção da
  reta mais o ruído perpendicular em cada eixo.
- O que sobra perpendicular à reta é a **inicialização congelada**:
  correlação 0,96 entre a coordenada perpendicular na época 1 e na 600, o
  desvio só caiu de 0,092 pra 0,053 (λ = 1e-4 encolhe devagar); zero
  correlação com r (Pearson −0,03) e nenhuma tendência suave (R² de uma
  parábola 0,002). O decoder nunca precisou da segunda direção, então o
  gradiente nela é ~0 e ela fica onde o sorteio a deixou.
- Pra GUI (Explore 2-D, dois sliders): cada slider mexe em parte ao longo
  da reta (muda o raio) e em parte perpendicular (o decoder deve ignorar, a
  conferir); o eixo "raio" é λ₀ − 1,2·λ₁, não nenhum dos dois sozinho.
- Próximo passo, se interessar: medir a sensibilidade do decoder à direção
  perpendicular (varrer ±0,1 perpendicular à reta a partir de um código do
  meio e ver se o SDF muda). Se não mudar, o d extra é de fato morto.

## Placa furo livre com d = 3: a mesma receita em 134 formas, 4 h (28/09)

Pedido: treinar `plate_hole_2d_xyr` (centro **e** raio variam, 134 placas) com
**três** componentes latentes, com a planilha do run d = 2
(`NN_Training_Hyperparameters_plate2d_r_only_d2_8x256_4h.xlsx`), em 4 h.

- **O relógio muda uma linha da receita.** Época = cada forma uma vez, então
  134 / 4 = 33 passos por época contra 10 no `r_only`. Medido: d1 à noite
  9,6 s/época (40 formas); d2 agora, 13,2; o `n134_d2` da GUI (8 × 256, 600
  épocas, de dia com a máquina em uso) 52 s/época, 36 no mínimo. Logo 134
  formas → 32–45 s/época com o laptop ocioso; as 600 épocas da planilha
  dariam 5,5–7,5 h e não cabem. Ficou **300 épocas, lr caindo pela metade a
  cada 75** (a mesma proporção 600/150), 9 900 passos de otimizador contra
  6 000 do run de 600 épocas do `r_only`. `--max-hours 4` segue como rede de
  segurança: se a máquina estiver mais lenta, para no prazo e o
  `metadata.json` diz em que época.
- **Planilha:** `docs/hyperparameters/NN_Training_Hyperparameters_plate2d_xyr_d3_8x256_4h.xlsx`
  é a do d = 2 com quatro células da coluna B mudadas — geometrias 134,
  dimensão latente 3, épocas 300, intervalo de decaimento 75 — e nada mais.
  σ = √(0,01 · 3) = 0,173, igual ao `preset_plate2d_xyr_d3`. Teste:
  `test_the_xyr_d3_sheet_is_the_same_recipe_sized_for_134_shapes`.
- **Código compartilhado subiu pra `structsept/app/unattended.py`:** log em
  arquivo + `.pid`, espera até a hora, `KeepAwake`, `prepare_run` genérico
  (dataset + dict de hiperparâmetros), o vigia do prazo, leitura dos códigos
  e do split. Segundo script precisando da mesma coisa → sobe pra
  `structsept/` (regra do `docs/structure.md`); `train_plate_r_only_4h.py`
  ficou só com a receita e a checagem do raio, mesma CLI. Testes em
  `tests/test_app_unattended.py`.
- **Script novo:** `experiments/train_plate_xyr_4h.py`. Além das flags do
  irmão: `--after <pid ou .pid>` espera outro processo terminar antes de
  treinar (encadear runs sem adivinhar a hora em que o anterior acaba; os
  dois scripts ganharam isso), `--decay-interval` (default: épocas / 4) e
  `--check-only --run X`, que roda só a comparação do fim num run já treinado.
- **Comparação do fim (`code_vs_params.csv/json/png` no diretório do run):**
  Pearson e Spearman de cada componente com cada parâmetro (tabela d × 3),
  o **ajuste afim (x_c, y_c, r) = W·λ + b** por mínimos quadrados com R² e
  RMSE (em unidades de projeto) de cada parâmetro — a comparação por
  regressão que o item de 23/09 pedia e ainda não existia —, o ajuste inverso
  (quão linear é cada componente nos parâmetros) e a variância dos códigos ao
  longo dos eixos principais. O `.png` é um grid: cada código contra cada
  parâmetro, e embaixo o ajustado contra o verdadeiro.
- **Já rodei essa checagem no run d = 2 da GUI**
  (`runs/plate_hole_2d_n134_d2_20260928_1338`, mesmo dataset byte a byte,
  8 × 256, 600 épocas, loss final 0,0085): código 0 ↔ `y_c` (Pearson 0,87),
  código 1 ↔ `x_c` (0,87), e **`r` não está no código: R² 0,009** (x_c 0,76,
  y_c 0,79; variância 53 / 47 % nos dois eixos). Com 25 placas o d = 2 tinha
  guardado `y_c` e largado `x_c`; com 134 guardou o centro e largou o raio.
  Faz sentido: mover o furo muda o SDF na placa inteira, mudar o raio muda
  uma faixa em volta dele — com dois números a rede fica com os dois que
  mais pesam na loss. É o argumento pro d = 3.
- **Lançado 28/09 ~23:55**, escondido (`Start-Process -WindowStyle Hidden`),
  `--after outputs/logs/plate2d_r_only_d2_8x256_4h.pid --max-hours 4`:
  espera o run d = 2 acabar (~01:25 no ritmo atual; no máximo 03:12, que é o
  prazo dele) e treina até 4 h. Previsão: 300 épocas em 2,7–3,7 h → pronto
  entre ~04:00 e ~05:15. Log em `outputs/logs/plate2d_xyr_d3_8x256_4h.log`,
  PID no `.pid` ao lado. Antes de lançar: `--epochs 1` num run de teste
  (78 s com o d2 rodando junto) e `--check-only` no run acima.
- **O que esperar:** (a) R² ≈ 1 nos três parâmetros e três eixos com
  variância parecida — o código é uma re-rotulagem afim de (x_c, y_c, r),
  e a aba Explore 2-D com três sliders varre centro e raio; (b) `x_c` e `y_c`
  altos e `r` baixo de novo — a terceira dimensão sobrou pra ruído e o raio
  precisa de outra coisa (mais épocas, λ menor, ou o clamp de 0,1 esconde
  os furos pequenos, r de 0,07); (c) Spearman alto e R² afim baixo — os
  parâmetros estão lá mas curvos; olhar o grid de dispersão antes de julgar.

**Resultado (29/09):** esperou o d = 2 acabar (01:09) e rodou de 01:09 a
03:56 — 2 h 47 min pras 300 épocas (33,4 s por época, bem no meio da
previsão), parou sozinho, sem bater no prazo. Loss final 0,0064 — abaixo
dos 0,0085 do d = 2 da GUI com o dobro de épocas. Saiu um misto de (a) e (c),
um parâmetro por vez (`runs/plate2d_xyr_d3_8x256_4h/code_vs_params.png`):

| parâmetro | R² afim | RMSE (un. projeto) | R² quadrático | onde está |
|---|---|---|---|---|
| `x_c` | **0,988** | 0,025 | 0,990 | código 1 (Pearson −0,85), linear |
| `r` | **0,857** | 0,025 | 0,905 | código 0 (Pearson +0,81), convexo: achatado nos furos pequenos, sobe rápido acima de r ≈ 0,2 |
| `y_c` | 0,741 | 0,114 | 0,786 | espalhado nos códigos 1 e 2 (Pearson +0,46 / −0,56), dobrado |

- **O raio entrou.** Era o que o d = 2 tinha largado (R² 0,009); com a terceira
  dimensão ele é o segundo parâmetro mais bem guardado, e a variância dos
  códigos fica 49 / 28 / 23 % nos três eixos — a rede usa as três direções.
- **`y_c` é o problema, e só na metade de baixo.** RMSE 0,164 pras placas com
  y_c < 0,45 contra 0,051 pras de cima; as cinco piores são furos perto da
  borda inferior (y_c 0,12–0,16) que o ajuste põe em 0,42–0,46 — o código
  não distingue "furo embaixo" de "furo no meio". Não é só curvatura: o
  ajuste quadrático quase não melhora (0,786). Como a placa é simétrica em
  x e y, a diferença entre x_c (perfeito) e y_c é acidente de treino — a
  direção que os códigos alinharam cedo —, não geometria.
- **Próximo passo natural:** olhar na aba Explore 2-D (três sliders) se o
  furo desce até a borda inferior ou trava no meio; se travar, os candidatos
  são mais épocas (a loss ainda oscilava 0,006–0,007 no fim) ou σ inicial
  maior pra espalhar os códigos antes do λ puxar. Um segundo seed diria se a
  dobra em y_c se repete.

## GUI: editar runs e decoders, tabela maior, dropdown do Explore 2-D (29/09)

Três pedidos feitos, todos em `tests/test_app_runs.py`, e uma pendência:

- **Editar um run — ou um decoder "trained here", que é a mesma pasta.** Na
  tabela de runs: botões *Open*, *Edit...*, *Load hyperparameters*,
  *Delete...* e *Refresh*, mais o menu do botão direito (e F2 / Delete na
  linha selecionada). *Edit...* abre uma janelinha
  (`structsept/app/run_editor.py`) com o nome e as notas do run: o nome é a
  pasta em `runs/`, renomear renomeia o decoder em todo lugar (a tabela e os
  dois pickers releem o disco na hora); as notas são o `Description` que o
  trainer já guarda no `specs.json` — ele só carrega a string, então run
  pronto continua carregando e preset esperando treina igual — e viram a
  última coluna da tabela. *Load hyperparameters* é o "Start from" sem
  procurar na lista. *Delete...* pergunta antes, recusa pasta sem
  `specs.json` e não tem volta. O run que está treinando agora fica travado
  pros três. O mesmo *Edit...* está no picker das duas abas Explore; fica
  cinza nos decoders da biblioteca, que não são runs.
- **Tabela de runs maior:** ficou com a metade larga da aba (3:2 em vez de
  2:3), 18 linhas em vez de 10, e as colunas de texto (run, notes) crescem
  com a janela; as numéricas não.
- **Dropdown do Explore 2-D (e do Explore) na largura do nome mais longo**
  (`widgets.combo_width`, entre 44 e 96 caracteres). Era 40 fixo e cortava
  `plate_hole_2d_n134_d2_20260928_1338 - d=2, 134 shapes (trained here)`.

### Pendência: duas GUIs ao mesmo tempo

Já funciona hoje, sem mexer em nada: cada janela é um processo com o seu
próprio Tk e o seu próprio estado (o `busy` é por janela, então dá pra
explorar numa enquanto a outra treina), o lock do atalho tem 16 vagas e o log
do launcher abre em modo append — dois cliques no atalho, ou dois
`uv run python -m structsept.app.main`. O que elas dividem é o disco,
`runs/` acima de tudo, e é aí que uma pode atrapalhar a outra:

- run que termina numa janela só aparece na outra depois de *Refresh* (as
  listas releem o disco nos eventos do próprio app, não num timer);
- mesmo nome de run nas duas = colisão: o nome automático tem resolução de
  minuto (`<dataset>_d<d>_<aaaammdd_hhmm>`), duas janelas no mesmo dataset no
  mesmo minuto escreveriam na mesma pasta — digitar o nome resolve;
- renomear/apagar na janela A um run que a B está treinando: o Windows recusa
  enquanto um arquivo dele estiver aberto, senão quebra o run da B no próximo
  checkpoint (`run_editor.training_now` só enxerga o run da própria janela);
- dois treinos dividem a CPU: o torch usa todos os núcleos e o tempo por
  época dos dois sobe (foi por isso que os scripts de 4 h rodaram um depois do
  outro).

Pra ficar "sem se influenciar" de verdade faltam duas coisas baratas
(~30 linhas): sufixo de segundos + PID no nome automático, e um timer de
refresh nas listas. Não fiz; fica na fila.

## Os buracos do "Latent coverage" e o dataset de 166 formas (30/09)

Pergunta: o que são as faixas bege do painel *Latent coverage* no decoder
`plate_hole_2d_xyr_d3_ep800`? São o maior trecho de cada eixo latente sem
nenhum código treinado, pintado quando passa de 12 % da faixa
(`viz.GAP_FRACTION`). Medido nos três runs d = 3 (códigos quase idênticos,
mesma seed):

| eixo | buraco | fração | forma sozinha do outro lado | vizinha mais próxima |
|---|---|---|---|---|
| λ₁ | +0,27 .. +0,47 | 29 % | furo centrado `r = 0,45` (extremo) | `r = 0,342` em (0,57; 0,50) |
| λ₃ | −0,31 .. −0,18 | 23 % | canto (0,88; 0,88), `r = 0,07` (extremo) | (0,866; 0,834), `r = 0,07` |
| λ₂ | +0,26 .. +0,29 | 5 % | canto (0,12; 0,88) | — (não pinta) |

- **O de λ₁ é buraco nos dados.** λ₁ carrega o raio (Pearson 0,82) e o
  dataset não tem furo entre 0,342 e 0,45: só 8 das 134 formas têm r > 0,25,
  a metade de cima da faixa de raio. Furo grande exige centro no meio da placa
  **e** `t` perto de 1 — um cantinho do cubo unitário —, e `--t-power` não
  resolve porque não puxa o centro pro meio.
- **O de λ₃ não é.** A vizinha do canto está a 0,05 no centro do furo, mesmo
  raio; e não é inicialização congelada (o código do canto saiu de +0,01 na
  época 1 pra −0,31 na 200 e ficou). É o espaço latente esticando perto do
  canto. Mais dados não garantem fechar.
- **`ep800` parou na época 680** (`latest.pth`; não tem `800.pth` nem
  `training_summary.json`). O run completo de 800 com a mesma receita é o
  `plate_hole_2d_xyr_d3_20260930_0124` (7 h 47, loss 0,0058).

O que foi feito:

- **`--large-holes N --large-holes-from R`** no `datagen.make_plate_hole`
  (`PlateHoleSpace.sample_large_holes` + `HoleParameters.extended`): N furos
  a mais com o **raio** uniforme em [R, r_max] (Sobol, um raio por fatia de
  1/N) e o centro uniforme no retângulo que aquele raio deixa — a seção da
  pirâmide naquela altura. Os três parâmetros continuam variando. Vão
  **depois** do sorteio principal, com seed + 1; como a seed das amostras é
  por índice, as primeiras formas saem byte a byte iguais às do dataset sem a
  flag. O manifesto grava em `parameter_sampling.large_holes`.
- **Dataset `plate_hole_2d_xyr_n166`** (100 MB): as 134 de sempre + 32 com r
  em [0,25; 0,45]. Maior salto de raio: 0,017 (era 0,108); formas com
  r > 0,25: 40 (eram 8). Conferido: os 134 `.npz` idênticos aos do `xyr`,
  auditoria da GUI sem problemas.

  ```
  uv run python -m datagen.make_plate_hole --dim 2 --n-uniform 25000 --n-band 25000 --large-holes 32 --large-holes-from 0.25 --name plate_hole_2d_xyr_n166 --plot
  ```
- **`experiments/train_plate_xyr_n166.py`**: copia o `specs.json` do run de
  referência (`--like`, default `ep800`) chave por chave, troca só split e
  descrição, imprime quais chaves diferem e se recusa a treinar se a receita
  mudar. Sem prazo. No fim mede a cobertura (a tabela acima) do run novo e da
  referência → `latent_coverage.json`; `--check-only` faz só isso.
- **Lançado 30/09 17:28**, escondido, `runs/plate_hole_2d_xyr_n166_d3_ep800`,
  800 épocas. Teste de 2 épocas antes: 58 s/época com o laptop em uso (166
  formas = 41 passos); à noite deve ir a ~43. Previsão: pronto entre ~03:00 e
  ~06:30 de 01/10. Log em `outputs/logs/plate_hole_2d_xyr_n166_d3_ep800.log`,
  PID no `.pid` ao lado.
- **O que esperar:** λ₁ sem faixa bege (os 16 raios novos acima de 0,35
  preenchem o trecho); λ₃ provavelmente continua com a dela.

## Placa com quatro furos triangulares: o dataset `plate_tri_2d_h` (30/09)

Segunda família de placa. Quadrado unitário, quatro triângulos isósceles, um
por quarto da placa: base paralela à borda mais próxima, ponta apontando pro
centro — sobra uma moldura e um X. Os centros (meio da altura do triângulo)
ficam fixos em 0,25 / 0,75. Único parâmetro: a altura `h`, igual nos quatro,
de **0,05 a 0,3**. Base = 2h (ponta de 90°, como no desenho de referência; os
braços do X ficam com largura constante).

As contas, no quadrado unitário:

- parede entre base e borda = folga entre ponta e centro = `1/4 − h/2`.
  Em h = 0,3 dá 0,1 — a mesma margem do furo redondo; em h = 0,05, 0,225.
  É isso que fixa `h_max = 1/2 − 2·margem = 0,3`.
- braço do X (largura transversal) = `(1/2 − h)/√2` → 0,141 em h_max.
  Nunca é o limitante.
- `h_min = 0,05` foi dado à mão: 1,6 células em N = 32 (o furo redondo pedia
  ≥ 4). O dataset não liga, o campo é exato; malhar as menores pra FEM vai
  precisar N ≥ 80.

O que foi feito:

- **`datagen/plate_tri_params.py`** — `PlateTriSpace(size, margin, h_min)`,
  `sample()` = varredura uniforme como o `sample_radius` do furo,
  `triangle_vertices()`. **`datagen/plate_tri_sdf.py`** — SDF exato
  `max(placa, −min dos 4 triângulos)` com a fórmula exata de distância a
  triângulo; frame, extrusão e config de amostragem são importados do módulo
  do furo, não copiados. Já aceita quatro alturas (ordem top, bottom, left,
  right) pra família seguinte. **`datagen/make_plate_tri.py`** — mesma CLI do
  `make_plate_hole`. **`datagen/preview.py`** — o `preview.png` que os dois
  scripts agora compartilham. Testes em `tests/test_datagen.py`: campo 2-D
  contra força bruta (sinal por `matplotlib.Path`, implementação
  independente), 3-D extrudado, banda nas arestas, layout dos arquivos,
  auditoria da GUI em 3-D.
- **Dataset `plate_tri_2d_h`** (24 MB): 40 alturas de 0,05 a 0,3 em passo
  uniforme, 25k uniformes + 25k banda por forma (50 % da banda nas 12
  arestas), mesmo frame do furo (pad 0,1, escala 1,8). Fração dentro
  51–69 %. `params.csv` tem `name, h, t, base, wall, arm`; `preview.png` e
  `parameters.png` na pasta.

  ```
  uv run python -m datagen.make_plate_tri --dim 2 --n-uniform 25000 --n-band 25000 --name plate_tri_2d_h --plot
  ```

- Pra "vários tipos de dados" depois: só falta um sampler de quatro alturas
  no `plate_tri_params` e colunas a mais no `params.csv` — a geometria já
  está pronta.

## Triângulos com d = 1: preset e treino de 8 min (30/09, 23:52)

Pedido: "adiciona o preset e lança o treino com d = 1".

- **Preset `runs/preset_plate2d_tri_h_d1`** — terceira linha do
  `experiments/make_plate_presets.py`, mesma receita dos outros dois (4 × 64
  ReLU, 1500 épocas, lr caindo pela metade em 500 e 1000, 4096 amostras por
  forma, 5 formas por batch → 8 passos por época, σ = 0,1), dataset
  `plate_tri_2d_h`, d = 1. O script reescreve os três presets; os dois antigos
  saem iguais.
- **`experiments/train_plate_tri_d1.py`** — copia o `specs.json` do preset
  chave por chave (só a descrição muda; imprime o que difere e se recusa se
  a receita mudasse), treina sem prazo, grava `metadata.json` e no fim
  compara o código aprendido com a coluna `h` do `params.csv` →
  `code_vs_h.csv` / `code_vs_h.png` na pasta do run (`--check-only` faz só a
  comparação). `--start-at`, `--after`, `--epochs N --run smoke` como os
  outros.
- **Duas funções foram pra `structsept/app/unattended.py`** porque ganharam
  um segundo chamador: `prepare_run_like` (copiar a receita de outro run;
  era o `prepare_run` do `train_plate_xyr_n166.py`) e `code_vs_parameter`
  (código × parâmetro gerador, qualquer coluna; era o
  `compare_code_with_radius` do `train_plate_r_only_4h.py`). Os dois scripts
  viraram wrappers finos; testes novos em `tests/test_app_unattended.py`
  (19 passam). Fumaça: tri com 2 épocas e n166 com 1 época, os dois ok.
- **Lançado 30/09 23:52:56**, escondido (`Start-Process -WindowStyle
  Hidden`), PID 20252, run `runs/plate_tri_2d_h_d1_4x64`. Log em
  `outputs/logs/plate_tri_2d_h_d1_4x64.log`, PID no `.pid` ao lado. Deve
  levar ~8–10 min.
- **O que esperar:** como no furo só-raio (Pearson −1,000, monotônico), o
  código deve sair uma reta contra `h`. Se não sair, o suspeito é o
  triângulo pequeno (h = 0,05 é uma feição de 0,1 de base) que o decoder
  4 × 64 pode borrar.

**Resultado (01/10, 00:13):** 1500 épocas em 20,6 min (mais que os ~8 min
estimados: os testes e os runs de fumaça rodaram junto no começo). O código
é uma **reta contra `h`**: Pearson −1,000, Spearman −1,000, monotônico, de
+0,111 (h = 0,05) a −0,162 (h = 0,3) — o mesmo resultado do furo só-raio, com
o triângulo pequeno incluído. `code_vs_h.png` / `.csv` e
`training_summary.json` na pasta do run.

## Triângulos com altura E largura livres: `plate_tri_2d_hw`, d = 2 (01/10)

Pedido: "test data + o test itself com o width mudando também, até 0,6
independente da altura — ou faça a escolha que achar melhor; latente 2".

- **A largura não pode ser 0,6 em qualquer altura.** O triângulo de cima e o
  da direita são imagens espelhadas pela diagonal da placa, então a distância
  entre eles é 2× a distância do triângulo à diagonal, atingida num vértice:
  `√2 · min(1/4 + (h − w)/2, 1/4 − h/2)`. Com w = 0,6 e h = 0,05 dá negativo
  — os dois se sobrepõem. A regra da margem 0,1 vira `w ≤ h + 0,5 − √2·0,1 =
  h + 0,359`. Escolha: **`w_max(h) = min(0,6; h + 0,359)`** — o 0,6 do
  pedido vale a partir de h ≈ 0,24; abaixo manda o ligamento. `w_min = 0,1`
  (a menor base da família só-altura). O conjunto viável em (h, w) é um
  trapézio; como no furo, sorteia-se num quadrado unitário `(t_h, t_w)` e
  mapeia-se (`from_unit`/`to_unit`), então o latente prescrito continua uma
  caixa. A família só-altura é a reta w = 2h dentro dessa. Conferido nos
  testes: fórmula do braço contra força bruta (distância mínima entre as
  duas linhas de contorno), em 4 pares (h, w).
- **Código:** `PlateTriSpace` ganhou `w_min`, `w_cap`, `max_width(h)`,
  `from_unit`/`to_unit`, `sample_hw()`; `clearances()` agora devolve 4
  linhas (wall, tip, arm, side) e aceita `w`. `TriParameters` ganhou `w`
  opcional (`None` = base 2h) — nomes `tri_h…_w…`, `params.csv` com
  `name,h,w,t_h,t_w,w_max,wall,arm,side`. `plate_tri_sdf` troca `base_ratio`
  por `bases` (largura explícita). `make_plate_tri --free-width`. Os 40 `.npz`
  do `plate_tri_2d_h` saem **byte a byte iguais** com o código novo
  (conferido). `code_vs_parameter` ganhou o R² do ajuste linear do parâmetro
  sobre os componentes (a base latente é arbitrária, então é isso, não
  componente a componente, que diz se o parâmetro está no código).
- **Dataset `plate_tri_2d_hw`** (32 MB): Sobol 128 + 5 extremos (cantos e
  centro do quadrado unitário: mínimo, agulha, lâmina, o desenho, meio) =
  133 formas, 25k + 25k por forma. Dentro 52–69 %.

  ```
  uv run python -m datagen.make_plate_tri --dim 2 --free-width --n-uniform 25000 --n-band 25000 --name plate_tri_2d_hw --plot
  ```
- **Preset `runs/preset_plate2d_tri_hw_d2`** (mesma receita, d = 2,
  σ = 0,141; 133/5 → 3 formas sobram por época). Launcher unificado
  `experiments/train_plate_tri.py --family h|hw` (substitui o
  `train_plate_tri_d1.py`); no fim faz `code_vs_h` e `code_vs_w`, cada um
  com o R² do ajuste linear. **Treino ainda não lançado** — depende do ok.
- **O que esperar:** os dois R² perto de 1 (h e w recuperados por
  combinações lineares dos dois componentes), com os eixos aprendidos numa
  orientação qualquer. Se w sair com R² baixo, o suspeito é a largura ser
  uma feição mais fraca no campo que a altura.

**Resultado (01/10, 08:43 → 09:33, 49,9 min — laptop em uso):** saiu o
esperado. Ajuste linear de `h` sobre (λ₀, λ₁): **R² = 1,000**; de `w`:
**R² = 0,996**. O plano latente é o trapézio (h, w) girado e cisalhado — no
`code_vs_h.png` a cor (h) varia ao longo de uma diagonal do plano, no
`code_vs_w.png` ao longo da outra. Componente a componente nada é
monotônico (λ₀ × h: Pearson 0,68; λ₁ × h: 0,74; λ₀ × w: 0,88; λ₁ × w: −0,47),
exatamente porque a base é arbitrária; 60,7 % da variância no primeiro eixo
principal (dois parâmetros, então não é mais uma reta). Ou seja: com dois
parâmetros geradores e d = 2, o decoder recupera os dois — a direção de `h`
no plano é `linear_fit_direction` do `code_vs_h.csv`, a de `w` a do outro.
Pra GUI (Explore 2-D): nenhum slider sozinho é "altura" ou "largura"; os
eixos úteis são essas duas direções.

**O run `plate_hole_2d_xyr_n166_d3_ep800` morreu.** Descoberto ao lançar
este: PID 14644 não existe mais, `latest.pth` gravado pela última vez às
18:51 na **época 80 de 800**, log sem erro nenhum (só as linhas de
partida). Causa desconhecida (a máquina não reiniciou). Opções: relançar do
zero com `--force` (encadeado com `--after outputs/logs/plate_tri_2d_h_d1_4x64.pid`),
ou continuar da época 80 — o `train_deep_sdf` da biblioteca aceita
`continue_from="latest"`, mas o `structsept.app.training.train` ainda não
expõe isso.

## Auditoria da rede contra o paper e o DeepSDFStruct (02/10)

Pedido: "dá uma olhada geral no código e vê se tá tudo certo com a NN", com o
paper e o repo como referência. Seis frentes — dados de treino, arquitetura,
trainer, hiperparâmetros, os runs treinados, uso do decoder — e cada achado
passou por um segundo passe independente que tentou derrubá-lo. **Nada foi
treinado**: só leitura de código e checkpoints e forward pass.

**O que está certo** (conferido, não suposto):

- **Dados:** o SDF das duas famílias é a distância exata, sinal certo
  (negativo no material), nas 672 formas em disco; o `.npz` e o split são o
  que o loader da biblioteca espera; índice do código = linha do `params.csv`.
- **Arquitetura:** o default do app é, chave por chave, o `NetworkSpecs` do
  `RoundCross` do autor (6 × 128, ReLU, dropout 0,2) e monta certo em 2-D;
  nada fixo em 3 coordenadas.
- **Treino:** a loss é a Eq. 7–8 do paper (L1 com clamp 0,1), Adam com
  5e-4 / 1e-3, o decaimento é a Eq. 10. Mesma seed = mesma rede, **bit a
  bit** (`ep800` e `0124` têm as 22 440 losses de batch idênticas).
- **Uso:** todo caminho de carga chama `eval()`; a entrada é [λ, x] igual ao
  treino; `T(x)` da biblioteca é a Eq. 18 exata; o gradiente chega nos pontos
  de controle (36/36 entradas não nulas).

**O achado que importa: o decoder d = 3 do furo livre desenha um segundo
furo.** `plate_hole_2d_xyr_d3_20260930_0124` (o run de 800 épocas, loss
0,0058): em **27 das 134 formas de treino**, decodificando com o próprio
código treinado, sai a placa com o furo certo **e mais um furo fantasma** do
mesmo tamanho, no meio do material. Figura:
`outputs/reconstruction/plate_hole_2d_xyr_d3_20260930_0124.png`.

- Não aparece em nenhum número de fim de run: loss plana, erro na banda
  0,0057, e a Eq. 38 do paper dá **0,0136 na média** (ao lado dos 0,0128 do
  paper) — mas **0,0032 na mediana**. A média é carregada pelas formas com
  fantasma (0,049 nelas, 0,0046 nas outras).
- É **uma** rede, não três: o run de 4 h (300 épocas), o `ep800` (680) e o
  `0124` (800) são a mesma trajetória, mesma seed. Fantasmas ao longo dela:
  0 na época 1, 1 na 50, 35 na 100, 20 na 150, 27 na 200, 24 na 400 e na
  600, 26 na 800. Apareceu cedo e as 500 épocas a mais não tiraram.
- É de um lado só: 21 das 32 formas com y_c < 0,30 têm fantasma, nenhuma com
  y_c > 0,55. É o "**`y_c` é o problema**" da entrada de 28/09 visto pelo
  outro lado — as cinco piores do ajuste afim são todas formas com fantasma.
  O código não separa "furo embaixo" de "furo no meio", e o decoder desenha
  os dois.
- Entre códigos treinados também: 3–7 de 30 interpolações saem com dois
  furos.
- **Causa: não sei** (precisaria treinar). Dois fatos: nunca foi tentada
  outra seed; e o n166 na época 80 tem 0 fantasmas onde a trajetória de 134
  já tinha 35 na época 100.
- `experiments/check_reconstruction.py` (novo, só forward pass) refaz isso
  pra qualquer run 2-D: Eq. 38, erro na banda, sinal errado, furos fantasma
  e furos faltando, csv + figura das 8 piores em `outputs/reconstruction/`.

| run | época | Eq. 38 média (mediana) | sinal errado | topologia |
|---|---|---|---|---|
| `plate_tri_2d_hw_d2_4x64` | 1500/1500 | 0,0011 (0,0011) | 0,53 % | ok |
| `plate_tri_2d_h_d1_4x64` | 1500/1500 | 0,0015 (0,0014) | 0,76 % | ok |
| `plate2d_r_only_d2_8x256_4h` | 600/600 | 0,0055 (0,0053) | 0,32 % | ok |
| `plate2d_r_only_d1_8x256_4h` | 600/600 | 0,0093 (0,0090) | 1,91 % | ok |
| `plate_hole_2d_n134_d2_20260928_1338` | 600/600 | 0,0111 (0,0051) | 1,89 % | um furo, mas inchado |
| `plate_hole_2d_xyr_n166_d3_ep800` | **80**/800 | 0,0078 (0,0075) | 0,90 % | 1 furo faltando |
| `plate2d_xyr_d3_8x256_4h` | 300/300 | 0,0146 (0,0032) | 0,96 % | **fantasma em 25** |
| `plate_hole_2d_xyr_d3_ep800` | **680**/800 | 0,0136 (0,0032) | 0,77 % | **fantasma em 26** |
| `plate_hole_2d_xyr_d3_20260930_0124` | 800/800 | 0,0136 (0,0032) | 0,76 % | **fantasma em 27** |
| `Test2` / `First_test` | 500 / 250 | 0,034 / 0,042 | 5–6 % | fantasmas (runs de fumaça) |

Os dois runs de triângulo e os dois `r_only` reproduzem as formas, bem
abaixo do erro do paper. O d = 2 nas três-parâmetros acerta a topologia mas
erra o raio em +20 % na média (o furo `r = 0,45` sai com 0,353).

**Receita — o intervalo 75 é resto da versão de 300 épocas.** A planilha
`..._plate2d_xyr_d3_8x256_4h.xlsx` diz hoje **600 épocas / intervalo 75**;
as outras duas dizem 600 / 150. O 75 entrou em 28/09 junto com as 300
épocas (a mesma proporção 600/150); as épocas voltaram pra 600 em 29/09 e o
intervalo ficou. Os runs de 800 herdaram: 10 metades, lr 5e-4 → 4,9e-7. A
partir da época ~450 o lr é < 1e-5: as últimas 350 épocas (3,5 h das 7,8 h)
baixaram a loss 3 %. Não medi se intervalo maior daria loss menor — só que
esse trecho não treina. O teste `test_the_xyr_d3_sheet_...` ainda espera 300
e **falha** (já falhava antes de hoje). Decisão do Tomás: 150 (a planilha
do supervisor), 200 (épocas/4 pra 800) ou manter.

**n166 não dá pra continuar.** O `continue_from` da biblioteca carrega os
códigos latentes e joga fora (`training.py:517`, `_ = ws.load_latent_vectors`):
continuaria com decoder da época 80 e códigos sorteados de novo. Corrige a
opção "continuar da época 80" da entrada de 01/10 — relançar é `--force`, do
zero, **~14 h** (62 s/época medidos no run que morreu, não os 43 previstos).

**Correção de uma conclusão antiga (28/09): "`r` não está no código".** O
R² afim de 0,009 se reproduz, mas mede "não é legível por um mapa linear",
não "ausente": regressão 3-vizinhos no espaço dos códigos recupera r com
R² 0,69, e o raio decodificado acompanha o verdadeiro (Spearman 0,86). O d = 2
guarda o raio numa superfície curva, e mal (+20 %); "d = 2 não basta" continua
valendo. E os números do run de 25 placas (−0,26 / 0,90 / 0,32) eram R² de
3-vizinhos, não afim — a frase comparava duas estatísticas diferentes.

**Paper × código de referência** (tabela em `docs/paper_context.md`): o
regularizador do código é a norma não elevada ao quadrado e o peso **sobe**
`min(1, época/100)`, enquanto a Eq. 9 impressa é `min(1, 1/n_época)` — conferi
na página renderizada. Pesa 0,3 % da loss, não muda nada. Fora do paper mas
no código: clip de gradiente em 1,0 (ativo nos checkpoints 375 a 1125 do
`plate_tri_2d_h_d1`, inativo nos 8 × 256), `CodeBound`, skip na camada 2,
weight norm, amostragem
metade dentro / metade fora. E os "16 000 pontos por passo" do paper são
**por forma** (160 000 por passo); a planilha divide pelo batch.

**Menores, pra saber:**

- **Dropout:** a loss do log é com dropout ligado; o decoder é usado sem. Nos
  8 × 256 isso desloca a borda externa da placa pra fora em eval — 0,013 no
  `r_only_d1` (do tamanho do erro do paper), 0,002 no d = 3. Os 4 × 64 sem
  dropout não têm isso.
- **Sliders do Explore 2-D:** a caixa min/max dos códigos tem muito espaço
  sem forma válida (37 de 100 pontos sorteados no d = 3), e o aviso de
  "código mais próximo" só dispara acima de 0,15 — 3 a 7 vezes o espaçamento
  dos códigos desses runs.
- **Malha:** a 10–14 cubos por célula a malha tem 6–12 % menos material que
  o SDF (o paper usa 20). `fem.tetrahedral_mesh` não é diferenciável (agora
  dito no docstring); o `create_3D_mesh(mesh_type="volume")` do teste de
  referência é, mas perde 17–32 % do volume.
- `plate_hole_2d_n134` e `plate_hole_2d_xyr` são o mesmo dataset, byte a byte.

**Corrigido hoje:**

- Tabela de runs, editor e os dois seletores do Explore mostram a época que o
  checkpoint tem (`80/800`, "epoch 80 of 800") — antes o n166 aparecia como
  800 épocas, treinado.
- *Volume fraction* do Explore: a grade de 32³ entrava em fase com o tiling
  (lia 0,00 num 8 × 8 × 8 com fração real 0,17; 0,16 / 0,18 alternando nos
  outros). Amostra sorteada dentro de cada célula, seed fixa: 0,173–0,176 em
  todos os tilings.
- Textos: docstring e ajudas do `hyperparams.py`, nota do import da planilha
  (a camada do skip não está na planilha e fica em 2; os runs 8 × 256 usaram
  4), comentário do R² em `unattended.py`, `LATENT_RANGE`, docstring do
  n166.
- 203 testes sem treino passam; o único que falha é o da planilha acima.

## Perguntas em aberto

- **(02/10)** O fantasma do d = 3 some com outra seed, com o intervalo de
  decaimento certo, ou com as 166 formas? Qual dos três rodar primeiro?
- **(02/10)** A linha "Points per training step" da planilha é por passo
  (como o app lê) ou por forma (como o paper e os specs do autor)?
- Qual espessura de placa faz sentido pro caso real do HiWi? Os 0,1 m foram
  chute meu.
- O furo é requisito do problema ou só exercício?
