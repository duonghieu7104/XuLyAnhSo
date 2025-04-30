# face_recognition_utils.py

import numpy as np
import cv2 as cv
import joblib
from PIL import Image

mydict = ['DaiLong', 'MinhHieu', 'VanThang', 'caothang']

# Load model 1 lần
svc = joblib.load('model/svc.pkl')
detector = cv.FaceDetectorYN.create(
    "model/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000
)
recognizer = cv.FaceRecognizerSF.create("model/face_recognition_sface_2021dec.onnx", "")

def recognize_from_ip_camera(ip_url, num_frames=1):
    cap = cv.VideoCapture(ip_url)
    if not cap.isOpened():
        return None, "Không thể mở camera IP!"

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    success, frame = cap.read()
    if not success:
        return None, "Không đọc được khung hình"

    faces = detector.detect(frame)
    if faces[1] is not None:
        face_align = recognizer.alignCrop(frame, faces[1][0])
        face_feature = recognizer.feature(face_align)
        test_predict = svc.predict(face_feature)
        name = mydict[test_predict[0]]
        cv.putText(frame, name, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        name = "Không phát hiện khuôn mặt"

    # Hiển thị FPS (bỏ tm cho đơn giản)
    cv.putText(frame, f'FPS: ~', (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Chuyển sang định dạng RGB cho Streamlit
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    return img_pil, name
