# Importamos librerias
import cv2
import numpy as np

# Cargamos el modelo COCO 80 clases
rcnn = cv2.dnn.readNetFromTensorflow('DNN/frozen_inference_graph_coco.pb',
                                     'DNN/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')

# Generamos los colores
colores = np.random.randint(0, 255, (81,3))

# Realizamos VidepCaptura
cap = cv2.VideoCapture(1)

while True:
    # Leemos el teclado
    t = cv2.waitKey(5)

    # Lectura de la VideoCaptura
    ret, frame = cap.read()
    alto, ancho, _ = frame.shape

    if t == 32:
        # Creamos una copia
        framec = frame.copy()
        muestra = frame.copy()
        # Alistamos nuestra imagen
        blob = cv2.dnn.blobFromImage(framec, swapRB=True)  # Swap: BGR -> RGB

        # Procesamos la imagen
        rcnn.setInput(blob)

        # Extraemos los Rect y Mascaras
        info, masks = rcnn.forward(["detection_out_final", "detection_masks"])

        # Extraemos la cantidad de objetos detectados
        contObject = info.shape[2]

        # Iteramos sobre los objetos detectados
        for i in range(contObject):
            # Extraemos los rectangulos de los objetos
            inf = info[0, 0, i]
            # print(inf)

            # Extraemos Clase
            clase = int(inf[1])
            print(clase)

            # Extraemos puntaje
            puntaje = inf[2]

            if puntaje < 0.7 or clase == 81:
                continue

            # Coordenadas del Rectangulos para deteccion de objetos
            x = int(inf[3] * ancho)
            y = int(inf[4] * alto)
            x2 = int(inf[5] * ancho)
            y2 = int(inf[6] * alto)

            # Extraemos el tamaño de los objetos
            tamobj = framec[y:y2, x:x2]
            tamalto, tamancho, _ = tamobj.shape
            # print(tamalto, tamancho)

            # Extraemos Mascara
            mask = masks[i, clase]
            mask = cv2.resize(mask, (tamancho, tamalto))
            # Establecemos un umbral
            _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            mask = np.array(mask, np.uint8)
            # print(mask.shape)

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
                cv2.fillPoly(tamobj, [cont], (r, g, b))
                # Dibujamos
                cv2.rectangle(framec, (x, y), (x2, y2), (r, g, b), 3)

                # print(cont)
            # Mostramos Mask
            cv2.imshow('MASK RCNN', framec)
            # Mostramos original
            cv2.imshow('MASCARA', muestra)

    # Mostramos FPS
    cv2.imshow('TIEMPO REAL', frame)
    # Si es ESC
    if t == 27:
        break

# Borramos la videocaptura
cap.release()
# Cerramos las ventanas
cv2.destroyAllWindows()
