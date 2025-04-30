import argparse
import os
import numpy as np
import cv2 as cv

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', '-if', type=str, default=r"C:\Users\LENOVO\Downloads\TuanDat")
parser.add_argument('--scale', '-sc', type=float, default=1.0)
parser.add_argument(
    '--face_detection_model', '-fd',
    type=str,
    default=r"C:\Users\LENOVO\Downloads\Nam3.2\XU_LY_ANH_SO\Do-an\NhanDangKhuonMatOnnx\model\face_detection_yunet_2023mar.onnx"
)
parser.add_argument(
    '--face_recognition_model', '-fr',
    type=str,
    default=r"C:\Users\LENOVO\Downloads\Nam3.2\XU_LY_ANH_SO\Do-an\NhanDangKhuonMatOnnx\model\face_recognition_sface_2021dec.onnx"
)
parser.add_argument('--score_threshold', type=float, default=0.9)
parser.add_argument('--nms_threshold', type=float, default=0.3)
parser.add_argument('--top_k', type=int, default=5000)

args = parser.parse_args()

save_dir = r"C:\Users\LENOVO\Downloads\Nam3.2\XU_LY_ANH_SO\Do-an\NhanDangKhuonMatOnnx\image\TuanDat"
os.makedirs(save_dir, exist_ok=True)

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(
                idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            for i in range(5):
                cv.circle(input, (coords[4+i*2], coords[5+i*2]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':
    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    recognizer = cv.FaceRecognizerSF.create(args.face_recognition_model, "")
    tm = cv.TickMeter()

    image_paths = [os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not image_paths:
        print("Không tìm thấy ảnh trong thư mục.")
        exit(1)

    # Đọc ảnh đầu tiên để đặt kích thước đầu vào cho detector
    first_image = cv.imread(image_paths[0])
    if first_image is None:
        print("Không thể đọc ảnh đầu tiên.")
        exit(1)
    h, w = first_image.shape[:2]
    detector.setInputSize([w, h])

    dem = 1
    for image_path in image_paths:
        frame = cv.imread(image_path)
        if frame is None:
            print(f"Không thể đọc ảnh: {image_path}")
            continue

        tm.start()
        faces = detector.detect(frame)
        tm.stop()

        visualize(frame, faces, tm.getFPS())
        cv.imshow('Live', frame)

        key = cv.waitKey(0)

        if key == 27:  # ESC để thoát
            break
        elif key == ord('s') or key == ord('S'):
            if faces[1] is not None:
                face_align = recognizer.alignCrop(frame, faces[1][0])
                file_name = os.path.join(save_dir, f"TuanDat_{dem:04d}.bmp")
                cv.imwrite(file_name, face_align)
                print(f"Đã lưu ảnh: {file_name}")
                dem += 1

    cv.destroyAllWindows()
