# Virtual Object Manipulator — Cubo 3D em Realidade Aumentada controlado por gestos

Projeto de visão computacional que projeta um cubo 3D sobre a imagem de uma webcam e permite manipulá-lo (mover, girar no eixo Z) usando gestos da mão, detectados via MediaPipe Hands. Construído como um projeto de aprendizado, aprofundando em geometria projetiva, calibração de câmera e design de interação por gestos.

## Demonstração


<img width="640" height="480" alt="demonstration" src="https://github.com/user-attachments/assets/89d99c78-f8f5-4499-87ce-dc79836251c6" />



## Funcionalidades

- **Renderização 3D calibrada**: o cubo é projetado sobre a cena real respeitando a perspectiva verdadeira da câmera.
- **Pegar e mover (translação em X, Y e Z)**: fechar a pinça (polegar + indicador) próximo ao cubo é possível movimentar o cubo nos três eixos.
- **Rotação (eixo Z)**: girar o pulso, mantendo a pinça fechada, rotaciona o cubo no plano da tela.

## Como funciona (arquitetura)

```
Webcam → Calibração (intrínseca + distorção)
              ↓
     MediaPipe Hands (21 landmarks)
              ↓
   Extração de gestos (ratio de pinça, ângulo do pulso,
   posição 2D, tamanho aparente da mão)
              ↓
   Máquina de estados (engatado / solto)
              ↓
   Atualização de rvec / tvec do cubo
              ↓
   cv2.projectPoints → desenho das faces e arestas
```

## Decisões técnicas

**Calibração de câmera validada por erro de reprojeção.** A câmera foi calibrada com `cv2.calibrateCamera` a partir de ~20 fotos de um tabuleiro de xadrez, validada com erro médio de reprojeção de ~0.07px. Os parâmetros intrínsecos e coeficientes de distorção são específicos da resolução de captura (640x480).

**Ratio de pinça normalizado pelo tamanho da mão.** Em vez de usar apenas a distância absoluta entre polegar e indicador (que varia com a distância da mão à câmera), a distância é dividida pela distância pulso→base do dedo médio. Isso torna o gesto de pinça robusto independente de quão perto ou longe a mão está da câmera.

**Movimento no eixo Z via tamanho aparente da mão, não via coordenada Z do MediaPipe.** A coordenada `z` retornada pelo MediaPipe é relativa e sem escala confiável, tornando-a inadequada para controle direto. Em vez disso, a variação do tamanho aparente da mão (em pixels) entre o momento do engate e o frame atual serve como proxy de aproximação/afastamento da câmera.

**Rotação limitada ao eixo Z.** Uma versão experimental calculava a normal do plano da palma (via produto vetorial entre três pontos da base da mão) para estimar rotação em X e Y. Essa abordagem depende da coordenada Z do MediaPipe, que se mostrou excessivamente ruidosa na prática, resultando em rotação instável e pouco controlável. A decisão final foi manter apenas rotação em Z (baseada no ângulo 2D do vetor pulso→base do dedo médio).

**Painter's algorithm para as faces do cubo.** As 6 faces são ordenadas por profundidade média (calculada a partir dos vértices já transformados por rotação e translação, antes da projeção 2D) e desenhadas da mais distante para a mais próxima, garantindo oclusão visual correta conforme o cubo gira.

## Limitações conhecidas

- Rotação restrita ao eixo Z (ver decisão técnica acima).
- Não há física real (gravidade, colisão, inércia).
- Tracking de mão pode ficar instável em ângulos muito oblíquos em relação à câmera, ou com iluminação desfavorável
- Calibração é específica da resolução 640x480 e da webcam usada (foi disponibilizado o código para a calibração da câmera)

## Requisitos

```
opencv-python
numpy
mediapipe
```

## Como executar

1. Calibre sua própria webcam rodando o script de calibração (gera `camera_matrix.npy` e `dist_coefficients.npy`) — use um tabuleiro de xadrez impresso, ~20 fotos em ângulos/distâncias variadas.
  Essa documentação ensina como essa etapa deve ser feita:
     https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_calibration.html  
3. Rode o script principal (`main.py`) com os arquivos `.npy` no mesmo diretório.
4. Posicione a mão em frente à câmera, feche a pinça próximo ao cubo para engatar, mova e gire.

## Estrutura do projeto

```
virtual-object-manipulator/
├── calibracao.py           # Script de calibração da câmera
├── main.py                 # Script principal (cubo + interação por gestos)
├── camera_matrix.npy       # Arquivo específico da calibração (Execute passo 1 caso planeje usar)
├── dist_coefficients.npy   # Arquivo específico da calibração (Execute passo 1 caso planeje usar)
└── README.md
```
