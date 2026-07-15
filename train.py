import cv2
import numpy as np
import os

dataset_path = "dataset"

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []

for file in os.listdir(dataset_path):
    path = os.path.join(dataset_path, file)

    gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    id = int(file.split(".")[1])

    faces.append(gray_img)
    ids.append(id)

recognizer.train(faces, np.array(ids))

recognizer.save("trainer.yml")

print("Training completed successfully ✅")