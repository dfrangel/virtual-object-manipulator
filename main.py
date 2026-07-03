import cv2
import numpy as np
import mediapipe as mp
import math

# Arquivos gerados a partir da calibração
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coefficients.npy')

L = 50  # Lado do cubo em mm
A = L / 2 # Metade do lado do cubo

# Define as coordenadas dos 8 vértices do cubo
cube_coords = np.array([
    [-A, -A, -A], [A, -A, -A], [A, A, -A], [-A, A, -A], # Vértices traseiros
    [-A, -A,  A], [A, -A,  A], [A, A,  A], [-A, A,  A]  # Vértices dianteiros
], dtype=np.float32)

# Define as 6 faces do cubo como listas de índices dos vértices (na ordem que formam o contorno)
faces_indices = [
    [0, 1, 2, 3],  # face de trás
    [4, 5, 6, 7],  # face da frente
    [0, 1, 5, 4],  # face de baixo
    [2, 3, 7, 6],  # face de cima
    [0, 3, 7, 4],  # face da esquerda
    [1, 2, 6, 5],  # face da direita
]

# Cores diferentes por face, só para facilitar visualizar a orientação do cubo
cores_faces = [
    (180, 50, 50),   # trás (Verde)
    (50, 180, 50),   # frente (Azul)
    (50, 50, 180),   # baixo (Ciano)
    (180, 180, 50),  # cima (Vermelho)
    (180, 50, 180),  # esquerda (Roxo)
    (50, 180, 180),  # direita (Amarelo)
]

# Estado do cubo (começa parado e de frente pra câmera a 25cm (250mm))
rvec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
tvec = np.array([0.0, 0.0, 250.0], dtype=np.float32)

# Chamada do MediaPipe com mp.solutions.hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cubo_segurado = False           # Booleano que confere se está segurando o cubo ou não
angulo_anterior = None          # Guarda o ângulo do frame passado, pra calcular o delta de rotação

raio_engate_dinamico = 0        # Delimite região em que a pinça deve estar para segurar o cubo
RATIO_FECHAR = 0.38             # Abaixo disso, considera pinça "fechada"
RATIO_ABRIR = 0.43              # Acima disso, considera pinça "aberta" (evita oscilação)

FATOR_TRANSLACAO_Z = 1.1        # Sensibilidade do movimento em Z
tamanho_mao_anterior = None     # Guarda o tamanho (pixels) da mão no ato de segurar

ESCALA_PX_PARA_MM = 0.6         # Escala de conversão: deslocamento em pixels na tela para mm no tvec

# Função de calcular a média (retorna o centro do cubo)
def calcular_centroide_2d(pontos_2d):
    return np.mean(pontos_2d, axis=0)

# Loop da Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a webcam.")
        break

    # Inverte para facilitar manipulação
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Projeção do cubo no estado atual
    pontos_2d, _ = cv2.projectPoints(cube_coords, rvec, tvec, camera_matrix, dist_coeffs)
    p = np.int32(pontos_2d).reshape(-1, 2)
    centro_cubo_2d = calcular_centroide_2d(p.astype(np.float32))

    # Converte o formeto RGB e processa a deteção da mão
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Se encontrar mãos
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Só 1 mão configurada

        # Facilita recall dos pontos principais
        pulso = hand_landmarks.landmark[0]
        base_medio = hand_landmarks.landmark[9]
        polegar = hand_landmarks.landmark[4]
        indicador = hand_landmarks.landmark[8]

        # Calcula abertura da pinça e normaliza pelo tamanho da mão
        tamanho_mao = math.hypot(base_medio.x - pulso.x, base_medio.y - pulso.y)
        tamanho_mao_px = math.hypot((base_medio.x - pulso.x) * w, (base_medio.y - pulso.y) * h)
        abertura_dedos = math.hypot(indicador.x - polegar.x, indicador.y - polegar.y)
        ratio_pinca = abertura_dedos / tamanho_mao

        # Calcula a posição da pinça em 2D
        pinca_x_px = (polegar.x + indicador.x) / 2 * w
        pinca_y_px = (polegar.y + indicador.y) / 2 * h
        pinca_pos_2d = np.array([pinca_x_px, pinca_y_px])

        # Calcula a rotação (eixo da linha do pulso -> base do dedo médio)
        dx = (base_medio.x - pulso.x) * w
        dy = (base_medio.y - pulso.y) * h
        angulo_atual = math.atan2(dy, dx)

        # Calcula a distancia da pinça até o cubo
        distancia_pinca_cubo = np.linalg.norm(pinca_pos_2d - centro_cubo_2d)

        # Calcula o "raio" do cubo na tela nesse frame, baseado na distância entre vértices opostos
        raio_cubo_atual_px = np.linalg.norm(p[0].astype(np.float32) - p[6].astype(np.float32)) / 2
        raio_engate_dinamico = raio_cubo_atual_px * 0.9 # faz o raio ser ligeiramente menor que o cubo

        if not cubo_segurado:
            # Só segura se a pinça estiver fechada e perto do cubo
            if ratio_pinca < RATIO_FECHAR and distancia_pinca_cubo < raio_engate_dinamico:
                cubo_segurado = True
                angulo_anterior = angulo_atual # Para calcular delta de rotação
                tamanho_mao_anterior = tamanho_mao_px # Para calcular movimento no eixo Z
        else:
            # Já está segurando só solta se abrir a pinça
            if ratio_pinca > RATIO_ABRIR:
                cubo_segurado = False
                angulo_anterior = None
                tamanho_mao_anterior = None


        # Calcula a translação e rotação do cubo
        if cubo_segurado:

            # Desloca o cubo na direção do movimento da pinça em relação ao centro atual do cubo
            delta_x_px = pinca_pos_2d[0] - centro_cubo_2d[0]
            delta_y_px = pinca_pos_2d[1] - centro_cubo_2d[1]

            tvec[0] += delta_x_px * ESCALA_PX_PARA_MM * 0.35
            tvec[1] += delta_y_px * ESCALA_PX_PARA_MM * 0.35
            # O fator 0.35 serve para suavizar o passo por frame

            # Move o cubo para frente ou trás se o tamanho da mão mudou
            if tamanho_mao_anterior is not None and tamanho_mao_anterior > 1e-3:
                # Se a variação > 1 a mão está aumentando (indo para frente) -> Z diminui
                # Se a variação < 1 a mão está diminuindo (indo para trás) -> Z aumenta
                variacao_escala = tamanho_mao_px / tamanho_mao_anterior
                delta_z = (1.0 - variacao_escala) * tvec[2] * FATOR_TRANSLACAO_Z
                tvec[2] += delta_z
                tamanho_mao_anterior = tamanho_mao_px  # atualiza a referência a cada frame

            # Impede o cubo de colapsar ou muito para frente ou muito para trás
            LIMITE_Z_MIN = 100
            LIMITE_Z_MAX = 600
            tvec[2] = np.clip(tvec[2], LIMITE_Z_MIN, LIMITE_Z_MAX)

            # Rotaciona o cubo conforme a variação do ângulo
            if angulo_anterior is not None:
                delta_angulo = angulo_atual - angulo_anterior
                rvec[2] += delta_angulo  # rotação no eixo Z da câmera (plano da tela)
                angulo_anterior = angulo_atual

        # Desenha os pontos da mão na tela
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    # Desenha os vértices atualizados na tela
    pontos_2d, _ = cv2.projectPoints(cube_coords, rvec, tvec, camera_matrix, dist_coeffs)
    p = np.int32(pontos_2d).reshape(-1, 2)

    # Calcula a pose 3D de cada vértice já transformada (rotação e translação)
    R, _ = cv2.Rodrigues(rvec)
    vertices_transformados = (R @ cube_coords.T).T + tvec

    # Profundidade média (Z) de cada face, pra saber a ordem de desenho
    profundidades_faces = []
    for indices in faces_indices:
        z_medio = np.mean([vertices_transformados[i][2] for i in indices])
        profundidades_faces.append(z_medio)

    # Ordena da face mais distante (Z maior) para a mais próxima (Z menor)
    ordem_desenho = np.argsort(profundidades_faces)[::-1]

    # É preciso usar um overlay para poder adicionar transparência
    # Desenha as faces preenchidas, na ordem correta (fundo primeiro)
    overlay = frame.copy()
    for idx_face in ordem_desenho:
        indices = faces_indices[idx_face]
        pontos_face = p[indices]
        cv2.fillPoly(overlay, [pontos_face], cores_faces[idx_face])
    cv2.circle(overlay, tuple(centro_cubo_2d.astype(int)), int(raio_engate_dinamico), (0, 0, 0), 1)

    # Adiciona a transparência e desenha um círculo indicando onde segurar
    alpha = 0.45
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Desenha as arestas
    for i, j in [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]:
        cv2.line(frame, tuple(p[i]), tuple(p[j]), (255, 255, 255), 1)

    cv2.imshow('Cubo AR + Interacao por Gestos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()