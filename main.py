import os
import numpy as np
from cvzone.PoseModule import PoseDetector
import cv2
import cvzone

# Inicializar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

detector = PoseDetector()

# Pasta para armazenar imagens de referência das pessoas
reference_folder = "references"

# Criar a pasta se não existir
if not os.path.exists(reference_folder):
    os.makedirs(reference_folder)

# Função para carregar imagens de referência das pessoas cadastradas
def load_reference_images():
    reference_images = {}
    for filename in os.listdir(reference_folder):
        name = os.path.splitext(filename)[0]
        image = cv2.imread(os.path.join(reference_folder, filename))
        reference_images[name] = image
    return reference_images

# Função para registrar uma nova pessoa
def register_person(name, image):
    cv2.imwrite(os.path.join(reference_folder, f"{name}.jpg"), image)

# Função para verificar se uma pessoa está cadastrada
def is_registered(pose_image, reference_images, threshold=0.6):
    for name, reference_image in reference_images.items():
        similarity = compare_images(pose_image, reference_image)
        if similarity >= threshold:
            return True, name
    return False, None

# Função para comparar duas imagens
def compare_images(image1, image2):
    difference = cv2.absdiff(image1, image2)
    similarity = np.sum(difference) / image1.size
    return 1 - similarity

# Carregar imagens de referência das pessoas cadastradas
reference_images = load_reference_images()

# Variáveis para controlar o estado de cadastro de uma pessoa
registering_person = False
new_person_name = ""

while True:
    # Capturar o frame da webcam
    ret, img = cap.read()

    # Verificar se o frame foi capturado corretamente
    if not ret:
        print("Erro ao capturar o frame da webcam.")
        break

    img = cv2.resize(img, (1280, 720))  # Redimensionar para as dimensões desejadas

    resultado = detector.findPose(img)
    pontos, bbox = detector.findPosition(img, draw=False)

    if len(pontos) >= 1:
        x, y, w, h = bbox['bbox']
        cabeca = pontos[0][1]
        joelho = pontos[26][1]
        tornozelo_esquerdo = pontos[27][1]
        tornozelo_direito = pontos[28][1]
        torso = pontos[11][1] - pontos[12][1]  # Calculando a inclinação do torso

        print(cabeca, joelho)
        diferenca = joelho - cabeca

        if diferenca <= 0:
            cvzone.putTextRect(img, 'QUEDA DETECTADA', (x, y - 80), scale=3, thickness=3, colorR=(0, 0, 255))
        elif torso > 45:  # Se a inclinação do torso for maior que 45 pixels, considerar como tropeço
            cvzone.putTextRect(img, 'TROPECO DETECTADO', (x, y - 80), scale=3, thickness=3, colorR=(0, 255, 0))
        elif abs(tornozelo_direito - tornozelo_esquerdo) > 50:  # Se a diferença entre tornozelos for maior que 50 pixels, considerar como escorregão
            cvzone.putTextRect(img, 'ESCORREGAO DETECTADO', (x, y - 80), scale=3, thickness=3, colorR=(255, 0, 0))

    # Verificar se uma nova pessoa está sendo registrada
    if registering_person:
        cv2.putText(img, 'DIGITE O NOME DA PESSOA E PRESSIONE ENTER:', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, new_person_name, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    else:
        cv2.putText(img, 'PRESSIONE "r" PARA INICIAR O REGISTRO DE UMA NOVA PESSOA', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('IMG', img)

    # Pressione 'r' para iniciar o registro de uma nova pessoa
    if cv2.waitKey(1) & 0xFF == ord('r'):
        registering_person = True
        new_person_name = ""

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Verificar se o usuário está digitando o nome da pessoa
    if registering_person:
        key = cv2.waitKey(0)
        if key == 13:  # Se pressionar Enter, terminar de registrar a pessoa
            registering_person = False
            if new_person_name:
                register_person(new_person_name, img)
        elif key == 27:  # Se pressionar Esc, cancelar o registro da pessoa
            registering_person = False
            new_person_name = ""
        elif key >= 32 and key <= 126:  # Se pressionar uma tecla de caractere ASCII
            new_person_name += chr(key)

# Liberar a captura de vídeo e fechar a janela
cap.release()
cv2.destroyAllWindows()
