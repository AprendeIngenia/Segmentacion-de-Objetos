# Importamos librerias
import cv2
import numpy as np

# Cargamos el modelo COCO 80 clases
rcnn = cv2.dnn.readNetFromTensorflow('DNN/frozen_inference_graph_coco.pb',
                                     'DNN/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')

# Leemos la imagen
img = cv2.imread('img2.jpeg')
alto, ancho, _ =  img.shape
#print(alto, ancho)

# Generamos los colores
colores = np.random.randint(0, 255, (80,3))

# Alistamos nuestra imagen
blob = cv2.dnn.blobFromImage(img, swapRB = True) # Swap: BGR -> RGB

# Procesamos la imagen
rcnn.setInput(blob)

# Extraemos los Rect y Mascaras
info, masks = rcnn.forward(["detection_out_final", "detection_masks"])

# Extraemos la cantidad de objetos detectados
contObject = info.shape[2]
#print(contObject)

# Iteramos sobre los objetos detectados
for i in range(contObject):
    # Extraemos los rectangulos de los objetos
    inf = info[0,0,i]
    #print(inf)

    # Extraemos Clase
    clase = int(inf[1])
    #print(clase)

    # Extraemos puntaje
    puntaje = inf[2]

    # Filtro
    if puntaje < 0.7:
        continue

    # Coordenadas del Rectangulos para deteccion de objetos
    x = int(inf[3] * ancho)
    y = int(inf[4] * alto)
    x2 = int(inf[5] * ancho)
    y2 = int(inf[6] * alto)

    # Extraemos el tamaño de los objetos
    tamobj = img[y:y2, x:x2]
    tamalto, tamancho, _ = tamobj.shape
    #print(tamalto, tamancho)

    # Extraemos Mascara
    mask = masks[i, clase]
    mask = cv2.resize(mask, (tamancho, tamalto))

    # Establecemos un umbral
    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    mask = np.array(mask, np.uint8)
    #print(mask.shape)

    # Extraemos coordenadas de la mascara
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Elegimos los colores
    color = colores[clase]
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    # Iteramos los contornos
    for cont in contornos:
        # Dibujamos mascara
        cv2.fillPoly(tamobj, [cont], (r,g,b))
        # Dibujamos
        cv2.rectangle(img, (x, y), (x2, y2), (r, g, b), 3)

        #print(cont)
    # Mostramos
    cv2.imshow('TAMANO OBJETO', tamobj)
    # # Mascara
    # cv2.imshow('MASCARA', mask)
    cv2.waitKey(0)

cv2.imshow('IMAGEN', img)
cv2.waitKey(0)