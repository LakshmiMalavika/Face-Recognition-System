# Face Recognition System

A real-time Face Recognition System built using Python and OpenCV. This project captures facial images, trains an LBPH face recognizer, and identifies registered users in real time using a webcam.

## Features

- Real-time face detection
- Face dataset collection
- Face recognition using LBPH algorithm
- Unknown face detection
- Webcam-based recognition

## Tech Stack

- Python
- OpenCV
- NumPy

## Project Structure

```
Face-Recognition-System/
│── face_detect.py
│── face_dataset.py
│── train.py
│── face_recognize.py
│── requirements.txt
│── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Collect Face Dataset

```bash
python face_dataset.py
```

### Step 2: Train the Model

```bash
python train.py
```

### Step 3: Run Face Recognition

```bash
python face_recognize.py
```

## Future Improvements

- Database integration
- Deep learning-based face recognition
- Web application using Flask
- Attendance management system

## Author

Lakshmi Malavika
