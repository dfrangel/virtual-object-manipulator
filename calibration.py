import numpy as np
import cv2
import os

#Variáveis de calibração do tabuleiro real
CANTOS_X = 5
CANTOS_Y = 8
TAMANHO_DO_QUADRADO_MM = 30.0

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Preparar pontos 3D do mundo real
objp = np.zeros((CANTOS_X * CANTOS_Y, 3), np.float32)
objp[:, :2] = np.mgrid[0:CANTOS_X, 0:CANTOS_Y].T.reshape(-1, 2)

#Multiplica pelo tamanho dos quadrados
objp *= TAMANHO_DO_QUADRADO_MM

objpoints = [] #Pontos 3D no espaço do mundo real
imgpoints = [] #Pontos 2D no plano da imagem

#Conecta com a webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
    exit()

print("========================================================")
print(" SCRIPT DE CAPTURA PARA CALIBRAÇÃO DA CÂMERA")
print("  [S] - Salva o frame atual (Capture em ângulos/distâncias diferentes)")
print("  [Q] - Sai do loop e calcula a calibração final")
print("========================================================")

contador_capturas = 0

while True:
    ret_cap, img = cap.read()
    if not ret_cap:
        print("Erro ao receber o frame da câmera. Encerrando...")
        break

    img = cv2.flip(img, 1)

    #Cópia do frame para não desenhar linhas por cima do que será salvo
    img_renderizacao = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret_tabuleiro, corners = cv2.findChessboardCorners(gray, (CANTOS_X, CANTOS_Y), None)

    corners2 = None
    #Se o tabuleiro foi encontrado, refina e desenha os cantos
    if ret_tabuleiro == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img_renderizacao, (CANTOS_X, CANTOS_Y), corners2, ret_tabuleiro)
    
    texto_status = f"Capturas salvas: {contador_capturas} (Recomendado: >15)"
    cv2.putText(img_renderizacao, texto_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    cv2.imshow('Calibracao da Camera - Webcam', img_renderizacao)
    
    key = cv2.waitKey(1) & 0xFF
    
    #Tecla 'S' para salvar
    if key == ord('s'):
        if ret_tabuleiro == True and corners2 is not None:
            objpoints.append(objp)
            imgpoints.append(corners2)
            contador_capturas += 1
            print(f"📸 Frame #{contador_capturas} salvo com sucesso!")
            
            # Correção do efeito de Flash visual na tela ao salvar
            flash = img.copy()
            cv2.imshow('Calibracao da Camera - Webcam', cv2.bitwise_not(flash))
            cv2.waitKey(100)
        else:
            print("❌ O tabuleiro não está visível ou não foi detectado corretamente. Posicione-o melhor!")

    #Tecla 'Q' para sair e processar
    elif key == ord('q'):
        if contador_capturas < 10:
            print(f"⚠️ Atenção: Você coletou apenas {contador_capturas} frames. O ideal são pelo menos 15 para precisão.")
        break

#Libera a webcam e fecha as janelas abertas
cap.release()
cv2.destroyAllWindows()

#Calibração final
if contador_capturas > 0:
    print("\nCalculando matrizes da câmera... (Por favor, aguarde)")
    
    #Executa o algoritmo de calibração do OpenCV
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if ret:
        print("SUCESSO! Câmera perfeitamente calibrada.\n")
        print("1. MATRIZ INTRÍNSECA DA CÂMERA (Matriz de Projeção):\n")
        print(mtx)
        print("\n2. COEFICIENTES DE DISTORÇÃO DA LENTE:")
        print(dist)


        #Calcula o erro médio de reprojeção
        erro_total = 0
        for i in range(len(objpoints)):
            imgpoints_reprojetados, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], mtx, dist
            )
            erro = cv2.norm(imgpoints[i], imgpoints_reprojetados, cv2.NORM_L2) / len(imgpoints_reprojetados)
            erro_total += erro

        erro_medio = erro_total / len(objpoints)
        print(f"\n3. ERRO MÉDIO DE REPROJEÇÃO: {erro_medio:.4f} pixels")

        if erro_medio < 0.5:
            print("Excelente calibração!")
        elif erro_medio < 1.0:
            print("Calibração aceitável.")
        else:
            print("Erro alto.")
        
        # Salva as matrizes geradas em arquivos do numpy (.npy)
        np.save('camera_matrix.npy', mtx)
        np.save('dist_coefficients.npy', dist)
else:
    print("\nNenhum frame foi coletado. O processo de calibração foi abortado.")