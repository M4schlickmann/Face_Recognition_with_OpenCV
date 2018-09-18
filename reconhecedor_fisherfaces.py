import cv2

detectorFace = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
reconhecedor = cv2.face.FisherFaceRecognizer_create()
reconhecedor.read('classificadorFisher.yml')
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_DUPLEX
camera =cv2.VideoCapture(1)


while(True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(50, 50))

    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (95, 255, 15), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        nome = ""
        if id == 1:
            nome = "Marcelo"
        elif id == 2:
            nome = "Luiz"
        else:
            nome = "Diego"
        cv2.putText(imagem, nome, (x,y + (a+50)), font, 1.5, (95, 255, 15))
        cv2.putText(imagem, str(confianca), (x,y + (a+80)), font, 1, (15, 12, 52))

    cv2.imshow('Face', imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindow()