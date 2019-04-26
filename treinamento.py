import cv2
import os
import numpy as np

##precisao do treinamento com mos filtros#
eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50, threshold=1)
fisherface = cv2.face.FisherFaceRecognizer_create(num_components=4, threshold=2700)
lbph = cv2.face.LBPHFaceRecognizer_create()

#percorre todo o caminho da aplicação para fazer o reconecimento das imagens#
def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    #print(caminhos)
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        #print(id)
        ids.append(id)
        faces.append(imagemFace)
        cv2.imshow('Face', imagemFace)
        cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = getImagemComId()
#armazena todas as faces com id's únicas

#processo de treinamenro para reconhecimento das faces
print('Treinando')
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

#utilização do filtro fisherface como padrão
fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print('Treinamento realizado')
