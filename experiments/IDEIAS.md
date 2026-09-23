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

## Perguntas em aberto

- Qual espessura de placa faz sentido pro caso real do HiWi? Os 0,1 m foram
  chute meu.
- O furo é requisito do problema ou só exercício?
