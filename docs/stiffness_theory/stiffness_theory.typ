// Da rede neural à matriz de rigidez -- o que acontece em
// experiments/plate_with_hole_stiffness.py + structsept/fem.py, explicado com a teoria.
// Recompilar (da raiz do repo, sem instalar nada no ambiente):
//   uv run --with typst python -c "import typst; typst.compile('docs/stiffness_theory/stiffness_theory.typ', output='docs/stiffness_theory.pdf')"
// As figuras em figures/ saem de make_figures.py mais os renders de
// structsept/plate_with_hole.py e structsept/fem.py.

#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1", number-align: center)
#set text(font: "New Computer Modern", size: 10.5pt, lang: "pt")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show raw: set text(font: "DejaVu Sans Mono", size: 8.6pt)
#show raw.where(block: true): it => block(
  fill: luma(247), stroke: 0.5pt + luma(200), inset: 8pt, radius: 3pt, width: 100%, it,
)
#show heading.where(level: 1): it => { v(0.9em); it; v(0.35em) }
#show heading.where(level: 2): it => { v(0.5em); it; v(0.2em) }
#show link: underline

// referência a equação sem o prefixo "Equação": só "(n)"
#let eqr(l) = ref(l, supplement: none)

// caixa "no código": liga cada pedaço de teoria à linha do script que o implementa
#let nocodigo(body) = block(
  fill: rgb("#eef3f8"), stroke: (left: 2.2pt + rgb("#3b6ea5")),
  inset: (x: 10pt, y: 7pt), radius: 2pt, width: 100%, above: 0.9em, below: 0.9em,
)[#set par(justify: false)
  #text(weight: "bold", fill: rgb("#3b6ea5"))[No código] #h(0.6em) #body]

// caixa de atenção
#let atencao(body) = block(
  fill: rgb("#fdf3e7"), stroke: (left: 2.2pt + rgb("#c96a1b")),
  inset: (x: 10pt, y: 7pt), radius: 2pt, width: 100%, above: 0.9em, below: 0.9em,
)[#set par(justify: false)
  #text(weight: "bold", fill: rgb("#c96a1b"))[Atenção] #h(0.6em) #body]

// ---------------------------------------------------------------------------
#align(center)[
  #text(size: 19pt, weight: "bold")[Da rede neural à matriz de rigidez]
  #v(0.3em)
  #text(size: 12.5pt)[O que acontece em `plate_with_hole_stiffness.py`, explicado com a teoria]
  #v(0.8em)
  #text(size: 10pt, fill: luma(90))[
    Projeto Struct_Sept, HiWi, TU Wien #sym.dot.c 8 de setembro de 2026 \
    Referência: Kofler, Giritsch, Elgeti, _Structural optimization of lattice structures using deep neural networks as geometry representation_, Graphical Models 142 (2025)
  ]
]
#v(1.2em)

#block(inset: (x: 1.2em))[
  *Resumo.* O script pega uma placa de treliça (_lattice_) com um furo, cuja geometria não é uma malha nem um campo de densidade, mas uma _função_: a distância com sinal até a superfície, calculada por uma rede neural. Ele transforma essa função em triângulos, os triângulos em tetraedros, e os tetraedros na matriz de rigidez global $bold(K)$ de um problema de elasticidade linear. E para aí. Este documento acompanha cada uma dessas etapas com a teoria por trás: o que é uma SDF, como o DeepSDF a representa, por que a treliça precisa de uma função de transformação, o que o FlexiCubes e o tetgen fazem, como a $bold(k)_e$ de um tetraedro sai da forma fraca, e o que a $bold(K)$ resultante tem de ter para estar certa. Todos os números citados foram medidos com o próprio script.
]

#v(0.5em)
#outline(title: "Conteúdo", indent: 1.2em, depth: 2)
#pagebreak()

// ===========================================================================
= Visão geral: o que entra e o que sai

O pipeline do artigo tem seis etapas. As duas primeiras (amostrar distâncias e treinar a rede) já foram feitas uma vez e os pesos estão salvos. O script executa as etapas 3 a 5, cortando a 5 no ponto em que a matriz de rigidez existe:

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (center, left, left),
    stroke: 0.5pt + luma(180),
    inset: 6pt,
    table.header([*Etapa*], [*O que acontece*], [*Função do script*]),
    [3], [A rede e a treliça viram uma única função $phi(bold(xi))$ no cubo $[0,1]^3$; o furo é subtraído; as bordas são fechadas.], [`plate_with_hole`],
    [4a], [FlexiCubes extrai a superfície $phi = 0$ como triângulos e o spline de deformação os leva para metros.], [`tetrahedral_mesh`],
    [4b], [tetgen preenche a superfície fechada com tetraedros.], [`tetrahedral_mesh`],
    [5a], [Cada tetraedro ganha uma matriz $bold(k)_e$ ($12 times 12$); todas são somadas em $bold(K)$.], [`build_solid`, `assemble_stiffness`],
    [5b], [_Não feito:_ condições de contorno, cargas, solução de $bold(K) bold(u) = bold(f)$.], [---],
    [6], [_Não feito:_ sensibilidades e atualização das variáveis de projeto (MMA).], [---],
  ),
  caption: [As etapas do pipeline e onde o script para.],
)

Um ponto que organiza tudo o que vem a seguir: *há dois sistemas de coordenadas*. A rede e a treliça vivem no cubo paramétrico $bold(xi) in [0,1]^3$. A placa física, de $1 times 1 times 0","1$ m, só aparece no fim, quando um spline leva os vértices da malha para metros. Sempre que uma quantidade é definida em metros (o furo, o material, a $bold(K)$) e outra no cubo (a rede, a treliça), alguma conversão tem de acontecer no meio, e o script tem uma classe só para isso.

// ===========================================================================
= A geometria como função: distância com sinal

== Definição

Seja $Omega subset RR^3$ um sólido e $Gamma = partial Omega$ a sua superfície. A *função distância com sinal* (SDF, _signed distance function_) é
$ phi(bold(x)) = cases(
  -min_(bold(y) in Gamma) norm(bold(x) - bold(y)) & "se" bold(x) in Omega "(dentro)",
  +min_(bold(y) in Gamma) norm(bold(x) - bold(y)) & "caso contrário (fora)",
) $ <eq:sdf>
A superfície é o conjunto de nível zero, $Gamma = {bold(x) : phi(bold(x)) = 0}$, e o material é $Omega = {phi < 0}$. Uma SDF exata satisfaz $norm(nabla phi) = 1$ em quase todo ponto: andar uma unidade em direção à superfície reduz a distância em exatamente uma unidade (@fig:sdf1d).

#figure(image("figures/sdf_1d.png", width: 78%), caption: [Uma SDF em uma dimensão. O sólido é o segmento entre os dois pontos vermelhos; dentro dele $phi < 0$. A inclinação é $plus.minus 1$ em todo lugar.]) <fig:sdf1d>

Por que representar geometria assim, e não por uma malha? Porque uma função pode ser avaliada em qualquer ponto, combinada com outras funções e, sobretudo, *derivada*: se $phi$ depende de parâmetros, a superfície $phi = 0$ também depende, de forma contínua. É isso que o otimizador precisa.

== Operações booleanas

Duas SDFs se combinam com `min` e `max`:
$ phi_(A union B) = min(phi_A, phi_B), quad
  phi_(A inter B) = max(phi_A, phi_B), quad
  phi_(A without B) = max(phi_A, -phi_B). $ <eq:bool>
A intuição para a diferença $A without B$: negar $phi_B$ troca dentro por fora, e a interseção com "o fora de $B$" é exatamente "$A$ menos $B$". Estas fórmulas dão o *conjunto de nível zero exato*, mas o valor longe da superfície é apenas um limite inferior da distância verdadeira (perto de cantos côncavos elas subestimam). Para extrair a superfície isso não importa.

== O furo

Um cilindro de raio $r$ com eixo $z$ passando por $(c_x, c_y)$ tem, no plano,
$ phi_"cil" (bold(x)) = sqrt((x - c_x)^2 + (y - c_y)^2) - r. $
O `CylinderSDF` da biblioteca é a versão 3D com tampas; o script o faz atravessar a placa inteira (de $z = -t$ a $z = 2t$) para que a tampa nunca caia dentro do material.

#nocodigo[
  `DifferenceSDF(lattice, ScaledSpaceSDF(hole, [L, W, t]))` implementa @eq:bool. A `ScaledSpaceSDF` existe por causa dos dois sistemas de coordenadas: o furo é descrito em metros, a treliça no cubo. A classe recebe o ponto paramétrico $bold(xi)$, estica-o para metros, $bold(x) = bold(S) bold(xi)$ com $bold(S) = "diag"(L, W, t)$, avalia o furo lá e devolve
  $ tilde(phi)_"furo" (bold(xi)) = frac(phi_"furo" (bold(S) bold(xi)), overline(S)), $
  com $overline(S)$ a média dos três fatores. O zero é exato; as magnitudes são apenas aproximadas quando a placa não é cúbica, porque um escalar não desfaz um estiramento anisotrópico. O FlexiCubes só usa as magnitudes para posicionar vértices, e ordem de grandeza certa basta.
]

// ===========================================================================
= A célula unitária como rede neural: DeepSDF

== A ideia

Uma rede DeepSDF é uma função $f_theta (bold(x), bold(lambda)) approx phi_bold(lambda)(bold(x))$ que, dado um ponto $bold(x) in [-1, 1]^3$ e um *vetor latente* $bold(lambda)$, devolve a distância com sinal até a superfície da geometria codificada por $bold(lambda)$. Os pesos $theta$ são treinados uma vez sobre uma família de formas; depois disso, mudar $bold(lambda)$ percorre continuamente essa família. É um espaço de projeto de baixa dimensão que só contém geometrias "parecidas com as de treino", ao contrário da otimização topológica por densidades, que pode produzir qualquer coisa.

O treino minimiza, sobre pontos amostrados $bold(x)_i$ e formas $j$,
$ min_(theta, {bold(lambda)_j}) sum_j sum_i abs(f_theta (bold(x)_i, bold(lambda)_j) - phi_j (bold(x)_i)). $
Cada forma de treino ganha o seu $bold(lambda)_j$ (inicializado de $cal(N)(0, 0","01)$), e os pesos são compartilhados.

== O modelo usado aqui

O script carrega `PretrainedModels.AnalyticRoundCross`, o caso de teste 1 do artigo: três cilindros perpendiculares de raio $r$, com *latente de dimensão 1* e $bold(lambda) = r$. Apesar do nome "decoder", este modelo em particular não passa pelos pesos: o `forward` calcula a fórmula fechada
$ f(bold(x), r) = min lr((norm(bold(x))_infinity, quad sqrt(y^2 + z^2) - r, quad sqrt(x^2 + z^2) - r, quad sqrt(x^2 + y^2) - r)), $ <eq:cross>
que é a união @eq:bool de três cilindros. O primeiro termo, $norm(bold(x))_infinity$, é só o valor inicial do `min`: positivo em todo ponto exceto a origem, não adiciona material, e garante que longe dos cilindros a distância cresça. É a geometria de referência com erro de reconstrução zero; os modelos `RoundCross`, `ChiAndCross` e `Primitives*` são redes de verdade com a mesma interface.#footnote[Verificado destruindo os pesos com ruído: a saída não muda. Vale a pena trocar para `RoundCross` e comparar as duas lado a lado, é o item 7 da fila em `experiments/IDEIAS.md`.]

#atencao[O latente só faz sentido na faixa em que a rede foi treinada, $r in [0","15, 0","75]$. Fora dela, uma rede de verdade devolve geometria sem significado. O script usa $r = 0","4$ constante.]

// ===========================================================================
= Do bloco à placa: empilhar células e graduar o latente

== A função de transformação

A rede conhece uma célula no cubo $[-1,1]^3$. A placa tem $t_x times t_y times t_z$ células, aqui $4 times 4 times 1$. A treliça é obtida com uma única função $T$, aplicada coordenada a coordenada, que mapeia o cubo paramétrico $[0,1]$ na entrada da rede $[-1,1]$ (eq. 18 do artigo):
$ T(x) = 4 abs(t_x x / 2 - floor(t_x x / 2 + 1/2)) - 1, quad phi_"treliça" (bold(xi)) = f_theta lr((T(xi_1), T(xi_2), T(xi_3), bold(lambda)(bold(xi)))). $ <eq:T>

#figure(image("figures/transformation.png", width: 80%), caption: [A função de transformação para $t_x = 4$. Cada célula vai de $-1$ a $1$; a seguinte volta de $1$ a $-1$. As linhas tracejadas são as interfaces entre células.]) <fig:T>

$T$ é uma onda triangular (@fig:T): dentro de uma célula ela é uma reta, e na interface entre células ela *espelha* em vez de saltar. É por isso que a SDF resultante é contínua mesmo para células assimétricas, que é o caso normal de geometria representada por rede: uma célula e a sua vizinha são imagens espelhadas uma da outra, então coincidem na interface. A alternativa da literatura (Chabra et al., _Deep Local Shapes_) usa uma transformação descontínua por célula mais uma função de peso; o artigo evita isso deliberadamente.

#nocodigo[`LatticeSDFStruct(tiling=[4, 4, 1], microtile=SDFfromDeepSDF(model), parametrization=Constant([0.4]))` implementa @eq:T. A função `transform` em `lattice_structure.py` é literalmente `2 * abs(t*x/2 - floor((t*x + 1)/2))`, reescalada para $[-1,1]$.]

== O campo latente

No artigo o latente varia no espaço, $bold(lambda)(bold(xi))$, e é isso que _gradua_ a treliça: barras grossas onde o esforço é grande, finas onde não é. A variação é dada por um B-spline com suporte local (eq. 21 do artigo),
$ bold(lambda)(bold(xi)) = sum_i N_i (bold(xi)) hat(bold(lambda))_i, $ <eq:latentfield>
e os pontos de controle $hat(bold(lambda))_i$ são *as variáveis de projeto* da otimização. Suporte local significa que mover um ponto de controle muda a treliça só numa vizinhança, sem perturbar o resto. O número de pontos de controle é independente do número de células.

Este script usa `Constant([0.4])`: $bold(lambda)(bold(xi)) = 0","4$ em toda a placa, todas as células iguais. É o caso degenerado de @eq:latentfield com um único ponto de controle, e é a única coisa que precisa mudar (trocar `Constant` por `SplineParametrization`) para o problema virar um problema de otimização.

== Fechar as bordas

Cortar a treliça pela fronteira do cubo deixa barras abertas: a superfície $phi = 0$ não é fechada, e uma superfície aberta não tem interior, logo não tem volume nem tetraedros. `CappedBorderSDF` faz a interseção da treliça com o próprio cubo, $phi = max(phi_"treliça", phi_"cubo")$ (@eq:bool), o que fecha cada barra com uma tampa plana na fronteira. A superfície resultante é estanque (_watertight_) e o script verifica isso antes de chamar o tetgen.

#figure(image("figures/hole_slices.png", width: 100%), caption: [Cortes da SDF final, já em metros. Esquerda: plano médio $z = 0","05$ m visto de cima; direita: seção $y = 0","5$ m. Azul é $phi < 0$ (material), vermelho $phi > 0$ (vazio); a linha preta é $phi = 0$. Vê-se o furo de raio $0","25$ m cortando as células centrais e as tampas retas na borda.]) <fig:slices>

// ===========================================================================
= Da função à malha

== Extração da superfície: FlexiCubes

O FEM precisa de uma malha. A superfície $phi = 0$ é extraída numa grade regular: $N$ pontos por célula e por eixo ($N = 10$, ou seja $41 times 41 times 11$ pontos no total), estendida 5 % para fora do cubo para que a superfície feche nas bordas. Em cada cubo da grade cujos vértices têm sinais diferentes de $phi$, a superfície passa por dentro.

O _marching cubes_ clássico coloca um vértice de malha em cada aresta com troca de sinal, por interpolação linear:
$ bold(x)_"e" = (phi_b bold(x)_a - phi_a bold(x)_b) / (phi_b - phi_a), $ <eq:interp>
o ponto da aresta $(a, b)$ onde a reta que liga $(bold(x)_a, phi_a)$ a $(bold(x)_b, phi_b)$ cruza zero. O *FlexiCubes* (Shen et al. 2023) é a versão _dual_ e diferenciável: cada cubo cortado recebe um único vértice, colocado numa média ponderada dos cruzamentos @eq:interp, e os vértices de cubos vizinhos são ligados por quadriláteros que depois viram triângulos. Os pesos são parâmetros extras que podem ser otimizados junto; aqui não são (`differentiate=False`).

Dois motivos para o FlexiCubes em vez do marching cubes: os triângulos saem bem formados (o marching cubes produz lascas, que são péssimas para o FEM) e, como @eq:interp é uma função diferenciável de $phi_a, phi_b$, *cada vértice da malha é uma função diferenciável dos valores da SDF*, logo do latente. Essa é a corrente que a otimização precisa manter intacta.

== Deformação livre: do cubo para metros

Os vértices extraídos estão em $[0,1]^3$. A biblioteca os leva para o domínio físico com um B-spline trilinear (_free-form deformation_),
$ overline(bold(x)) = sum_i N_i (bold(xi)) bold(P)_i, $
que para uma caixa se reduz a $overline(bold(x)) = (L xi_1, W xi_2, t xi_3)$. Aqui é só um estiramento anisotrópico, mas como é um `TorchSpline`, os pontos de controle $bold(P)_i$ podem depois entortar a placa em qualquer forma sem tocar no resto do código. No artigo eles *não* são variáveis de projeto.

== Preencher o interior: tetgen

A superfície fechada (8192 triângulos, 4096 vértices) vai para o tetgen, que constrói uma tetraedrização de Delaunay restrita: os triângulos da superfície são preservados como faces (opção `Y`) e o interior é preenchido inserindo *pontos de Steiner* onde for preciso para respeitar um critério de qualidade (opção `q`, que limita a razão raio-aresta). Saem 4667 nós (os 4096 da superfície mais 571 internos) e 16 015 tetraedros. O volume total dos tetraedros, $sum_e V_e = 0","020241$ m$""^3$, coincide com o volume fechado pela superfície em todos os dígitos impressos, que é o teste de que a tetraedrização não perdeu nem inventou material.

#figure(
  grid(columns: 2, gutter: 8pt,
    image("figures/tets_render.png", width: 100%),
    image("figures/tets_render_clipped.png", width: 100%)),
  caption: [A malha de tetraedros que entra no torch-fem, inteira (esquerda) e cortada pelo plano $y = 0","5$ m (direita). O furo de raio $0","25$ m remove as quatro células centrais; sobra um anel de doze.],
) <fig:tets>

#atencao[
  A biblioteca também oferece `create_3D_mesh(mesh_type="volume")`, que usa a tetraedrização interna do FlexiCubes e é o que o teste de otimização da biblioteca usa. Medido num cubo unitário maciço, esse caminho devolve tetraedros que somam apenas 0,68 a 0,70 do volume, em qualquer resolução, e o contorno da malha tem 13 034 triângulos em 5836 componentes (a superfície tem 1200 em uma). São milhares de cavidades internas. A matriz de rigidez de uma malha assim é a de um corpo com 30 % de vazios. Por isso o script usa o tetgen, cujo volume bate com o da superfície.
]

== Orientação

Antes de virar elemento finito, cada tetraedro precisa de orientação positiva. O volume com sinal
$ V_e = 1/6 (bold(x)_1 - bold(x)_0) dot lr([(bold(x)_2 - bold(x)_0) times (bold(x)_3 - bold(x)_0)]) $ <eq:vol>
é positivo se os quatro nós seguem a regra da mão direita. Se for negativo, trocar dois nós inverte o sinal sem mudar a geometria. O motivo de isso importar aparece em @eq:ke: o determinante do Jacobiano do elemento é $6V_e$, e um determinante negativo daria uma $bold(k)_e$ negativa. Tetraedros de volume nulo (quatro nós coplanares) têm Jacobiano singular e são descartados. Com a saída do tetgen os dois passos não fazem nada, mas custam nada e protegem contra surpresas.

// ===========================================================================
= Elementos finitos: da elasticidade à matriz de rigidez

== O problema contínuo

Um sólido elástico linear ocupando $overline(Omega)$ obedece, em cada ponto, ao equilíbrio, à cinemática de pequenas deformações e à lei de Hooke:
$ nabla dot bold(sigma) + bold(b) = bold(0), quad
  bold(epsilon) = 1/2 lr((nabla bold(u) + nabla bold(u)^T)), quad
  bold(sigma) = bold(C) : bold(epsilon). $ <eq:strong>
Para material isotrópico, $bold(sigma) = lambda_L "tr"(bold(epsilon)) bold(I) + 2 mu bold(epsilon)$, com as constantes de Lamé
$ lambda_L = (E nu) / ((1 + nu)(1 - 2 nu)), quad mu = E / (2(1 + nu)). $
Com aço, $E = 210$ GPa e $nu = 0","3$, vem $lambda_L approx 121$ GPa e $mu approx 81$ GPa. Em notação de Voigt, $bold(epsilon) = (epsilon_(x x), epsilon_(y y), epsilon_(z z), gamma_(x y), gamma_(y z), gamma_(x z))^T$ e $bold(C)$ é a matriz $6 times 6$
$ bold(C) = mat(delim: "[",
  lambda_L + 2 mu, lambda_L, lambda_L, 0, 0, 0;
  lambda_L, lambda_L + 2 mu, lambda_L, 0, 0, 0;
  lambda_L, lambda_L, lambda_L + 2 mu, 0, 0, 0;
  0, 0, 0, mu, 0, 0;
  0, 0, 0, 0, mu, 0;
  0, 0, 0, 0, 0, mu). $ <eq:C>

#nocodigo[`torchfem.materials.IsotropicElasticity3D(E=210e9, nu=0.3)` é @eq:C. A biblioteca a "vetoriza": uma cópia por elemento, o que permitiria material diferente em cada tetraedro.]

== A forma fraca

Multiplicando o equilíbrio por um deslocamento virtual $bold(v)$ (nulo onde $bold(u)$ é prescrito) e integrando por partes, chega-se ao princípio dos trabalhos virtuais: encontrar $bold(u)$ tal que, para todo $bold(v)$ admissível,
$ underbrace(integral_Omega bold(epsilon)(bold(v)) : bold(C) : bold(epsilon)(bold(u)) dif Omega, a(bold(u), bold(v)) "-- trabalho interno")
  = underbrace(integral_Omega bold(v) dot bold(b) dif Omega + integral_(Gamma_N) bold(v) dot bold(t) dif Gamma, ell(bold(v)) "-- trabalho externo"). $ <eq:weak>
A forma bilinear $a(dot, dot)$ é simétrica (porque $bold(C)$ é) e é ela que vira a matriz de rigidez. O lado direito vira o vetor de forças. A rigidez, portanto, *não depende das cargas nem dos apoios*: só de geometria e material. É por isso que o script pode montá-la sem ter decidido nada sobre condições de contorno.

== Discretização com tetraedros lineares

Dentro de um tetraedro de nós $bold(x)_1 dots bold(x)_4$, o deslocamento é interpolado linearmente,
$ bold(u)(bold(x)) = sum_(i=1)^4 N_i (bold(x)) bold(u)_i, $
com $N_i$ as *coordenadas baricêntricas*: $N_i = 1$ no nó $i$, $0$ nos outros, lineares em $bold(x)$ e somando 1. Cada $N_i = (a_i + b_i x + c_i y + d_i z) \/ (6 V_e)$, de modo que $nabla N_i = (b_i, c_i, d_i) \/ (6 V_e)$ é *constante* no elemento. A deformação de Voigt fica $bold(epsilon) = bold(B) bold(u)_e$, com $bold(u)_e in RR^12$ os 4 nós $times$ 3 componentes e
$ bold(B) = lr([bold(B)_1 space bold(B)_2 space bold(B)_3 space bold(B)_4]), quad
  bold(B)_i = 1 / (6 V_e) mat(delim: "[",
    b_i, 0, 0;
    0, c_i, 0;
    0, 0, d_i;
    c_i, b_i, 0;
    0, d_i, c_i;
    d_i, 0, b_i). $ <eq:B>
Substituindo em $a(bold(u), bold(v))$ restrito a um elemento, como $bold(B)$ e $bold(C)$ são constantes, a integral é trivial:
$ bold(k)_e = integral_(Omega_e) bold(B)^T bold(C) bold(B) dif Omega = V_e bold(B)^T bold(C) bold(B) quad (12 times 12). $ <eq:ke>
Um único ponto de integração é *exato* para este elemento. Em unidades: $bold(C)$ em Pa, $bold(B)$ em 1/m, $V_e$ em m$""^3$, logo $bold(k)_e$ em N/m, rigidez de mola.

#nocodigo[
  `solid.k0()` calcula @eq:ke para os 16 015 elementos de uma vez, tensor de forma `(16015, 12, 12)`. O torch-fem faz isso pelo caminho isoparamétrico: mapeia um tetraedro de referência para o físico, $bold(J) = partial bold(x) \/ partial bold(xi)$, obtém $nabla N_i = bold(J)^(-T) nabla_xi N_i$ e integra $w dot det bold(J) dot bold(B)^T bold(C) bold(B)$ com $w = 1\/6$ e $det bold(J) = 6 V_e$, o que dá exatamente $V_e bold(B)^T bold(C) bold(B)$. Este é o lugar onde um tetraedro invertido faria estrago: $det bold(J) < 0$ produziria $bold(k)_e$ negativa.
]

== Montagem

Os deslocamentos nodais do elemento são um recorte do vetor global: $bold(u)_e = bold(L)_e bold(u)$, com $bold(L)_e$ uma matriz booleana $12 times n$ que seleciona os graus de liberdade (GDL) $3 i + d$ dos quatro nós, $d in {0, 1, 2}$ para $x, y, z$. A energia total é a soma das energias dos elementos, e daí
$ bold(K) = sum_(e=1)^(n_e) bold(L)_e^T bold(k)_e bold(L)_e quad (n times n), quad n = 3 n_"nós" = 14 001. $ <eq:assembly>
Ninguém forma as $bold(L)_e$ de verdade: cada entrada $(i, j)$ de $bold(k)_e$ é *somada* na posição $("gdl"_i, "gdl"_j)$ de $bold(K)$ (_scatter-add_). Como nós compartilhados por vários elementos recebem contribuições de todos eles, a mesma posição é atingida várias vezes, e é a soma que faz a estrutura se comportar como um todo em vez de como 16 015 tetraedros soltos.

#nocodigo[
  `solid.assemble_matrix(k, con)` implementa @eq:assembly em formato esparso COO: gera as triplas (linha, coluna, valor) de todos os elementos, 16 015 $times$ 144 $=$ 2,3 milhões delas, e `coalesce()` soma as que caem na mesma posição, sobrando 488 007 entradas distintas. O argumento `con` lista GDL prescritos, cujas linhas e colunas são trocadas pela identidade; o script o passa vazio de propósito.
]



== Esparsidade

$K_(i j) != 0$ só se os GDL $i$ e $j$ pertencem a algum elemento em comum, ou seja, se os nós são vizinhos na malha. $bold(K)$ é, portanto, a matriz de adjacência da malha com blocos $3 times 3$ nas entradas. Na placa, cada linha tem em média 35 não-zeros (o nó e os seus cerca de 11 vizinhos, vezes 3), e a matriz tem 0,25 % de preenchimento. A @fig:spy mostra o padrão: a banda diagonal são os 4096 nós de superfície, numerados pelo FlexiCubes ao longo da grade (por isso vizinhos no espaço são vizinhos no índice); o bloco denso no canto são os 571 nós internos que o tetgen inseriu depois, sem ordem espacial.


#figure(image("figures/sparsity_doc.png", width: 46%), caption: [Padrão de esparsidade de $bold(K)$ ($14 001 times 14 001$, 488 007 não-zeros). Cada ponto preto é uma entrada não nula.]) <fig:spy>

// ===========================================================================
= O que a matriz de rigidez tem de satisfazer

Três propriedades seguem da teoria, e o script mede as três. Elas servem como testes porque cada uma falha de um jeito diferente.

== Simetria

$bold(K)^T = bold(K)$ porque $a(bold(u), bold(v)) = a(bold(v), bold(u))$, ou, algebricamente, porque $(bold(B)^T bold(C) bold(B))^T = bold(B)^T bold(C)^T bold(B)$ e $bold(C)$ é simétrica. Medido: $max abs(K_(i j) - K_(j i)) \/ max abs(K_(i j)) = 2 times 10^(-16)$, o épsilon da máquina em precisão dupla. Uma assimetria maior indicaria erro na montagem.

== Semidefinida positiva, com núcleo de dimensão seis

A energia de deformação $1/2 bold(u)^T bold(K) bold(u) = 1/2 integral bold(epsilon) : bold(C) : bold(epsilon) dif Omega >= 0$ para todo $bold(u)$, logo $bold(K)$ é semidefinida positiva e todos os elementos da diagonal são positivos (medido: mínimo $1","1 times 10^9$ N/m). A igualdade a zero ocorre exatamente quando $bold(epsilon) = 0$ em todo lugar, ou seja, para deslocamentos que *não deformam*: os movimentos de corpo rígido. Em 3D são seis:
$ bold(u) = bold(c) quad "(três translações)", quad quad bold(u) = bold(omega) times bold(x) quad "(três rotações infinitesimais)". $ <eq:rigid>
A rotação não deforma porque $nabla(bold(omega) times bold(x)) = [bold(omega)]_times$ é antissimétrica, e a parte simétrica de uma matriz antissimétrica é zero. Assim, sem apoios, $bold(K)$ é *singular* com núcleo de dimensão seis: $bold(K) bold(u)_"rígido" = bold(0)$.

O script constrói os seis vetores de @eq:rigid para todos os nós e mede $max abs(bold(K) bold(u)_"rígido") \/ max abs(K_(i j)) = 3","7 times 10^(-16)$. Este é o teste mais forte dos três: ele passa apenas se a geometria (as $bold(B)$), o material (a $bold(C)$) e a montagem (as posições de _scatter_) estiverem todos coerentes entre si. Um erro numa única $bold(B)$ já quebraria a rotação.

A @fig:spec mostra o mesmo fato pelo espectro, medido na placa maciça (menor, para caber numa diagonalização densa). Uma $bold(k)_e$ tem posto 6: seis autovalores em torno de $10^(-7)$ N/m e seis entre $10^8$ e $3 times 10^(10)$. A $bold(K)$ inteira ($2271 times 2271$) tem seis autovalores em torno de $10^(-4)$ N/m e o sétimo em $8","2 times 10^6$: um vão de dez ordens de grandeza que separa os modos rígidos dos modos elásticos.

#figure(image("figures/spectra.png", width: 92%), caption: [Autovalores em módulo, escala logarítmica. Esquerda: os 12 de uma matriz de elemento; direita: os 10 menores de $bold(K)$ da placa maciça. A faixa vermelha marca os seis modos de corpo rígido, numericamente zero.]) <fig:spec>

== O que os números da corrida dizem

#figure(
  table(
    columns: (1fr, auto, auto),
    align: (left, right, right),
    stroke: 0.5pt + luma(180),
    inset: 6pt,
    table.header([*Quantidade*], [*Placa treliça com furo*], [*Placa maciça com furo*]),
    [Triângulos na superfície], [8192], [1456],
    [Nós / tetraedros], [4667 / 16 015], [757 / 2478],
    [Graus de liberdade $n$], [14 001], [2271],
    [Não-zeros de $bold(K)$], [488 007 (0,25 %)], [78 147 (1,5 %)],
    [Maior entrada de $bold(K)$], [$1","4 times 10^(11)$ N/m], [$6","0 times 10^(11)$ N/m],
    [Assimetria relativa], [$2","2 times 10^(-16)$], [$3","0 times 10^(-16)$],
    [Menor diagonal], [$1","1 times 10^9$ N/m], [$1","1 times 10^(10)$ N/m],
    [$max abs(bold(K) bold(u)_"rígido")$, relativo], [$3","7 times 10^(-16)$], [$2","9 times 10^(-16)$],
    [Tempo total], [$approx$ 7 s], [$approx$ 5 s],
  ),
  caption: [Saída das duas configurações do `launch.json`, resolução 10, aço.],
)

A placa maciça tem entradas maiores porque, com a mesma resolução, os seus tetraedros são maiores e mais "gordos" na direção do plano: $bold(k)_e prop V_e \/ h^2$ para um elemento de tamanho $h$, e a treliça é feita de barras finas.

// ===========================================================================
= O que falta, e por que o script para aqui

Com $bold(K)$ montada, o caminho até a otimização do artigo é:

#[
#set par(justify: false)
+ *Condições de contorno de Dirichlet.* Prescrever $bold(u) = bold(0)$ nos nós de uma face (`solid.constraints[mask, :] = True`). O torch-fem troca as linhas e colunas desses GDL pela identidade, o que remove os seis modos rígidos e torna $bold(K)$ definida positiva, logo invertível.
+ *Cargas de Neumann.* O vetor $bold(f)$ de @eq:weak, distribuído pelos nós (`solid.forces[mask, d] = ...`).
+ *Solução.* $bold(K) bold(u) = bold(f)$ por um solver esparso direto (`solid.solve(method="spsolve")`). É o gargalo de custo: no artigo, 1,5 dos 1,8 minutos de cada iteração.
+ *Objetivo.* A flexibilidade (_compliance_) $J = bold(f)^T bold(u) = bold(u)^T bold(K) bold(u)$, o dobro da energia de deformação. Minimizá-la é maximizar a rigidez sob a carga dada.
+ *Restrição.* O volume $V = sum_e V_e$, direto de @eq:vol, tem de ficar abaixo de um alvo.
+ *Sensibilidades.* Como $J$ é auto-adjunto, $dif J \/ dif p = - bold(u)^T (partial bold(K) \/ partial p) bold(u)$ para qualquer parâmetro $p$. A cadeia inteira
  $ hat(bold(lambda)) arrow.r bold(lambda)(bold(xi)) arrow.r f_theta arrow.r phi arrow.r "vértices (FlexiCubes)" arrow.r overline(bold(x)) "(FFD)" arrow.r bold(K) arrow.r J $
  é diferenciável, e o `torch.autograd` a percorre de trás para a frente. É por isso que nada no laço pode sair do torch: um `.numpy()` ou `.detach()` no meio corta a corrente.
+ *Atualização.* O MMA (_Method of Moving Asymptotes_) move os pontos de controle $hat(bold(lambda))$ dentro da faixa $[0","15, 0","75]$ e o laço recomeça na etapa 3.
]

`DeepSDFStruct/tests/test_structural_optimization.py` faz tudo isso numa viga em balanço e é o modelo para o próximo script. A única ressalva: ele monta a malha pelo caminho com cavidades descrito na seção 5.3, então os seus números de flexibilidade não correspondem à geometria que a rede descreve.

// ===========================================================================
= Resumo em uma tabela

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, left),
    stroke: 0.5pt + luma(180),
    inset: 6pt,
    table.header([*Função*], [*Teoria*], [*Equação*]),
    [`plate_with_hole`], [SDF da célula (rede), empilhamento com $T(x)$, latente constante, furo por diferença booleana, tampas por interseção, spline de deformação.], [#eqr(<eq:sdf>), #eqr(<eq:bool>), #eqr(<eq:cross>), #eqr(<eq:T>)],
    [`tetrahedral_mesh`], [Extração do nível zero (FlexiCubes), mapa para metros (FFD), tetraedrização de Delaunay restrita (tetgen), orientação positiva.], [#eqr(<eq:interp>), #eqr(<eq:vol>)],
    [`build_solid`], [Lei de Hooke isotrópica em Voigt; definição do problema discreto.], [#eqr(<eq:strong>), #eqr(<eq:C>)],
    [`assemble_stiffness`], [Matriz de elemento do tetraedro linear e montagem por _scatter-add_.], [#eqr(<eq:B>), #eqr(<eq:ke>), #eqr(<eq:assembly>)],
    [`check_stiffness`], [Simetria, positividade da diagonal, núcleo de dimensão seis.], [#eqr(<eq:rigid>)],
    [`export`], [$bold(K)$ em CSR (`.npz`), malha em `.vtk`, padrão de esparsidade.], [---],
  ),
  caption: [Cada função do script e a teoria que ela implementa.],
)
