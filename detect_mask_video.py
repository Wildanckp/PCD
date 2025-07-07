# detect_mask_video.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model deteksi wajah
face_net = cv2.dnn.readNet("face_detector/deploy.prototxt",
                           "face_detector/res10_300x300_ssd_iter_140000.caffemodel")

# Load model deteksi masker
mask_model = load_model("training/model/mask_detector.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0) / 255.0

            (mask, withoutMask) = mask_model.predict(face)[0]
            label = "Masker" if mask > withoutMask else "Tanpa Masker"
            color = (0, 255, 0) if label == "Masker" else (0, 0, 255)

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imshow("Deteksi Masker", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
