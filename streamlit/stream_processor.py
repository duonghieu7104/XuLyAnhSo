import cv2
import numpy as np
import joblib
import os
import av

# Xác định đường dẫn tương đối tới thư mục chứa mô hình
current_dir = os.path.dirname(__file__)
model_dir = os.path.join(current_dir, 'model')  # Thư mục chứa .onnx và svc.pkl

# Tải mô hình SVM
svc = joblib.load(os.path.join(model_dir, 'svc.pkl'))
mydict = ['DaiLong' 'MinhHieu' 'MinhHoang' 'VanThang' 'caothang']

class VProcessor:
    def __init__(self):
        print("Initializing face detector and recognizer...")
        self.detector = cv2.FaceDetectorYN.create(
            os.path.join(model_dir, 'face_detection_yunet_2023mar.onnx'),
            "",
            (320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000
        )
        print("Face detector initialized.")

        self.recognizer = cv2.FaceRecognizerSF.create(
            os.path.join(model_dir, 'face_recognition_sface_2021dec.onnx'),
            ""
        )
        print("Face recognizer initialized.")

    def process_frame(self, img):
        # Cập nhật kích thước ảnh đầu vào
        self.detector.setInputSize((img.shape[1], img.shape[0]))
        print(f"Set input size to: {(img.shape[1], img.shape[0])}")

        faces = self.detector.detect(img)
        print(f"Detected faces: {faces[1]}")

        if faces[1] is not None:
            for face in faces[1]:
                face_align = self.recognizer.alignCrop(img, face)
                face_feature = self.recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]

                coords = face[:-1].astype(np.int32)
                cv2.rectangle(img, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
                cv2.putText(img, result, (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return img

    def recv(self, frame):
        print("Processing frame...")
        img = frame.to_ndarray(format="bgr24")
        img = self.process_frame(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
