import numpy as np
import os
import cv2
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

class IdentityMetadata():
    def __init__(self, base, name, file):
        self.base = base
        self.name = name
        self.file = file

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

def load_metadata(path):
    metadata = []
    for person_name in sorted(os.listdir(path)):
        person_dir = os.path.join(path, person_name)
        if not os.path.isdir(person_dir):
            continue
        for fname in sorted(os.listdir(person_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in ['.jpg', '.jpeg', '.bmp']:
                metadata.append(IdentityMetadata(path, person_name, fname))
    return np.array(metadata)

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

# Đường dẫn tới model và folder ảnh
image_folder = r"C:\Users\LENOVO\Downloads\Nam3.2\XU_LY_ANH_SO\Do-an\NhanDangKhuonMatOnnx\image"
face_det_model = r"C:\Users\LENOVO\Downloads\Nam3.2\XU_LY_ANH_SO\Do-an\NhanDangKhuonMatOnnx\model\face_detection_yunet_2023mar.onnx"
face_rec_model = r"C:\Users\LENOVO\Downloads\Nam3.2\XU_LY_ANH_SO\Do-an\NhanDangKhuonMatOnnx\model\face_recognition_sface_2021dec.onnx"

# Load metadata
metadata = load_metadata(image_folder)

# Load detector và recognizer
detector = cv2.FaceDetectorYN.create(
    face_det_model, "", (320, 320), 0.9, 0.3, 5000
)
recognizer = cv2.FaceRecognizerSF.create(face_rec_model, "")

# Trích xuất embedding
embedded = np.zeros((len(metadata), 128))

for i, m in enumerate(metadata):
    img = cv2.imread(m.image_path())
    if img is None:
        print(f"Không thể đọc ảnh: {m.image_path()}")
        continue

    detector.setInputSize((img.shape[1], img.shape[0]))
    faces = detector.detect(img)

    if faces[1] is None or len(faces[1]) == 0:
        print(f"Không phát hiện khuôn mặt trong ảnh: {m.image_path()}")
        continue

    aligned_face = recognizer.alignCrop(img, faces[1][0])
    face_feature = recognizer.feature(aligned_face)
    embedded[i] = face_feature

# Gán nhãn cho dữ liệu
targets = np.array([m.name for m in metadata])
encoder = LabelEncoder()
y = encoder.fit_transform(targets)

# Chia train/test
total = len(embedded)
train_idx = np.arange(total) % 5 != 0
test_idx = np.arange(total) % 5 == 0
X_train = embedded[train_idx]
X_test = embedded[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

# Huấn luyện mô hình
svc = LinearSVC()
svc.fit(X_train, y_train)
acc_svc = accuracy_score(y_test, svc.predict(X_test))
print(f'SVM accuracy: {acc_svc:.4f}')

# Lưu model
model_save_path = os.path.join(os.path.dirname(face_rec_model), 'svc.pkl')
joblib.dump(svc, model_save_path)
print(f"Đã lưu mô hình tại: {model_save_path}")

print("Các nhãn:", encoder.classes_)

