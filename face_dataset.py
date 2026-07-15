import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

# Ask user ID
face_id = input("Enter your ID (example: 1): ")

count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1

        # Save image in dataset folder
        cv2.imwrite(
            f"dataset/User.{face_id}.{count}.jpg",
            gray[y:y+h, x:x+w]
        )

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) == 13 or count >= 100:
        break

cap.release()
cv2.destroyAllWindows()