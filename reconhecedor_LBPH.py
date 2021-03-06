import cv2

detectorFace = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
reconhecedor = cv2.face.LBPHFaceRecognizer_create(2, 2, 15, 15, 70)
reconhecedor.read('classificadorLBPH.yml')
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera =cv2.VideoCapture(1)

#Converte a inmágem de entrada em escalka de cinza para aplicar os filtros de seleção
while(True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(50, 50))

    #identificação de faces com banco interno (possível fazer uma conecção com banco de dados) 
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        3
       #padrão para desenhar um retangulo envolta da face onde será feito o reconhecimento de faces
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        nome = ""
        if id == 1:
            nome = "Face_1"
        elif id == 2:
            nome = "Face_2"
        else:
            nome = "Face_3"

        cv2.putText(imagem, nome, (x,y + (a+30)), font, 2, (0, 0, 255))
        cv2.putText(imagem, str(confianca), (x,y + (a+50)), font, 1, (0, 0, 255))

    cv2.imshow('Face', imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindow()
