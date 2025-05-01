import streamlit as st
import os
import cv2
import numpy as np
import tempfile

from tkinter.filedialog import askopenfilename, asksaveasfilename
import chapter3 as c3
import chapter4 as c4
import chapter9 as c9

from face_recognition_utils import recognize_from_ip_camera
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoTransformerBase,
)
from video_processor import VideoProcessor
from stream_processor import VProcessor

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from keras import backend as K
from keras.models import Model
from keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dropout, Reshape, Dense,
    Bidirectional, LSTM
)

from ocr_utils import preprocess_image_gray, build_inference_model, ctc_decode, num_to_text

# === 1) Page configuration ===
st.set_page_config(
    page_title="Ứng dụng Xử lý Ảnh",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === 2) Custom CSS for contrast & sizing ===
st.markdown(
    """
    <style>
    .main .css-1lcbmhc p,
    .main .css-1lcbmhc span,
    .main .css-1lcbmhc h1,
    .main .css-1lcbmhc h2,
    .main .css-1lcbmhc h3 {
        color: #333333 !important;
    }
    .css-1d391kg { color: #222222 !important; }
    .sidebar .sidebar-content {
        background-image: linear-gradient(180deg,#2E86C1,#AED6F1);
        color:white;
    }
    .sidebar .css-1n544kv,
    .sidebar .css-1uvaat5 { color:white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# === 3) Sidebar with logo & menu ===
sidebar = st.sidebar
sidebar.image("logo.jpg", use_column_width=True)
sidebar.markdown("---")
menu = sidebar.radio(
    "Chọn chức năng:",
    (
        "Nhận diện khuôn mặt",
        "Nhận diện đối tượng",
        "Xử lý ảnh số",
        "Đọc chữ viết tay"
    )
)

st.title("🖼️ Ứng dụng xử lý hình ảnh")

# === 4) ICE servers & RTC config ===
ICE_SERVERS = {
    "iceServers": [
        {"urls": ["stun:stun.relay.metered.ca:80"]},
        {
            "urls": ["turn:global.relay.metered.ca:80"],
            "username": "8ea627684cd9d3f67e9a5618",
            "credential": "NK/rJapMisCsSdYh",
        },
        {
            "urls": ["turn:global.relay.metered.ca:443"],
            "username": "8ea627684cd9d3f67e9a5618",
            "credential": "NK/rJapMisCsSdYh",
        },
    ]
}
RTC_CONFIGURATION = RTCConfiguration(ICE_SERVERS)

def start_streaming_in_column(col, processor_factory):
    with col:
        webrtc_streamer(
            key="webrtc-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=processor_factory,
            async_processing=True,
        )

# === 5) Nhận diện khuôn mặt ===
if menu == "Nhận diện khuôn mặt":
    st.header("🔍 Nhận diện khuôn mặt")
    option = st.radio(
        "Chọn nguồn đầu vào:",
        ("Tải ảnh", "Tải video", "Stream Camera"),
    )

    if option == "Tải ảnh":
        uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg","jpeg","png","bmp"])
        if uploaded_file:
            data = uploaded_file.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Ảnh gốc", channels="BGR", width=400)
            processor = VideoProcessor()
            result_img, result_text = processor.process_frame(img)
            with col2:
                st.image(result_img, caption="Kết quả nhận diện", channels="BGR", width=400)
            if result_text:
                st.text(f"Nhận diện: {result_text}")

    elif option == "Tải video":
        uploaded_video = st.file_uploader("Tải video lên", type=["mp4","avi","mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            processor = VideoProcessor()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                out_frame, _ = processor.process_frame(frame)
                stframe.image(out_frame, channels="BGR", width=640)
            cap.release()

    elif option == "Stream Camera":
        st.subheader("📡 Streaming từ webcam (WebRTC)")
        col_live, _ = st.columns([1, 2])
        start_streaming_in_column(col_live, VProcessor)

# === 6) Nhận diện đối tượng ===
elif menu == "Nhận diện đối tượng":
    st.header("🧠 Nhận diện đối tượng")
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg","jpeg","png","bmp"])
    if uploaded_file:
        data = uploaded_file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        model_path = os.path.join("model", "best.onnx")
        try:
            model = YOLO(model_path)
        except Exception as e:
            st.error(f"Không thể tải mô hình: {e}")
        else:
            results = model.predict(img, conf=0.5, verbose=False)
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            confs = results[0].boxes.conf.tolist()
            names = model.names

            img_out = img.copy()
            annotator = Annotator(img_out)
            for box, cls, conf in zip(boxes, clss, confs):
                label = f"{names[int(cls)]} {conf:.2f}"
                annotator.box_label(box, label=label, txt_color=(255,0,0), color=(255,255,255))

            st.image(img_out, caption="Kết quả nhận diện", channels="BGR", width=600)

# === 7) Xử lý ảnh số ===
elif menu == "Xử lý ảnh số":
    st.header("🔢 Xử lý ảnh số")
    def load_image():
        file_path = st.file_uploader("Choose an image...", type=["jpg", "png", "tif", "bmp", "webp"])
        if file_path is not None:
        # Đọc ảnh và kiểm tra xem nó có hợp lệ không
           image = cv2.imdecode(np.frombuffer(file_path.read(), np.uint8), cv2.IMREAD_COLOR)
           if image is None:
            st.error("Error: Image could not be loaded.")
            return None
        st.image(image, channels="BGR", caption="Uploaded Image.", use_column_width=True)
        return image
    
        return None     

    def save_image(img):
       if st.button("Save Processed Image"):
           file_path = asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("BMP files", "*.bmp")])
           if file_path:
            cv2.imwrite(file_path, img)
            st.success(f"Image saved successfully as {file_path}")
            
            
    def chapter3_operations(img):
       st.subheader("Chapter 3 Operations")
       operation = st.selectbox("Choose an operation", [
        "Negative", "NegativeColor", "Logarit", "Power", "Piecewise Line",
        "Histogram", "Hist Equal", "Hist Equal Color", "Local Hist", "Hist Stat",
        "Smooth Box", "Smooth Gauss", "Median Filter", "Sharpening", 
        "Sharpening Mask", "Grandient"
    ])
    
       img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
       imgout = None  # Khởi tạo biến ảnh đầu ra
    
    # Áp dụng các bộ lọc theo lựa chọn của người dùng
       if operation == "Negative":
        imgout = c3.Negative(img_gray)
       elif operation == "NegativeColor":
        imgout = c3.NegativeColor(img)
       elif operation == "Logarit":
        imgout = c3.Logarit(img_gray)
       elif operation == "Power":
        imgout = c3.Power(img_gray)
       elif operation == "Piecewise Line":
        imgout = c3.PiecewiseLine(img_gray)
       elif operation == "Histogram":
        imgout = c3.Histogram(img_gray)
       elif operation == "Hist Equal":
        imgout = cv2.equalizeHist(img_gray)
       elif operation == "Hist Equal Color":
        imgout = c3.HistEqualColor(img)
       elif operation == "Local Hist":
        imgout = c3.LocalHist(img_gray)
       elif operation == "Hist Stat":
        imgout = c3.HistStat(img_gray)
       elif operation == "Smooth Box":
        imgout = cv2.boxFilter(img_gray, cv2.CV_8UC1, (21, 21))
       elif operation == "Smooth Gauss":
        imgout = cv2.GaussianBlur(img_gray, (43, 43), 7.0)
       elif operation == "Median Filter":
        imgout = cv2.medianBlur(img_gray, 5)
       elif operation == "Sharpening":
        imgout = c3.Sharpening(img_gray)
       elif operation == "Sharpening Mask":
        imgout = c3.SharpeningMask(img_gray)
       elif operation == "Grandient":
        imgout = c3.Grandient(img_gray)

    # Giảm kích thước hiển thị ảnh để dễ xem
       img_resized = cv2.resize(img, (300, 300))
       imgout_resized = cv2.resize(imgout, (300, 300))

    # Hiển thị hai ảnh cạnh nhau
       col1, col2 = st.columns(2)
       with col1:
           st.image(img_resized, caption="Ảnh Gốc", use_column_width=True)
       with col2:
           st.image(imgout_resized, caption="Ảnh Xử Lý", use_column_width=True)

       return imgout

    def chapter4_operations(img):
        st.subheader("Chapter 4 Operations")
        operation = st.selectbox("Choose an operation", ["Spectrum", "Remove Morie", "Remove Inference", "Create Motion"])
    
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển sang grayscale nếu cần
        imgout = None

        if operation == "Spectrum":
           imgout = c4.Spectrum(img_gray)  # Nếu Spectrum chỉ hoạt động với ảnh xám
        elif operation == "Remove Morie":
           imgout = c4.RemoveMorie(img_gray)  # Dùng ảnh xám để tránh lỗi unpacking
        elif operation == "Remove Inference":
           imgout = c4.RemoveInference(img_gray)  # Nếu cần grayscale
        elif operation == "Create Motion":
           imgout = c4.CreateMotion(img_gray)  # Nếu hàm cần ảnh xám

        if imgout is not None:
           img_resized = cv2.resize(img, (300, 300))
           imgout_resized = cv2.resize(imgout, (300, 300))

        # Hiển thị hai ảnh cạnh nhau
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_resized, caption="Ảnh Gốc", use_column_width=True)
        with col2:
            st.image(imgout_resized, caption="Ảnh Xử Lý", use_column_width=True)

        return imgout 
         
    def chapter9_operations(img):
        st.subheader("Chapter 9 Operations")
        operation = st.selectbox("Choose an operation", ["Erosion", "Dilation", "Boundary", "Contour"])

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám trước khi xử lý
        imgout = None

        if operation == "Erosion":
           imgout = c9.Erosion(img_gray)
        elif operation == "Dilation":
           imgout = c9.Dilation(img_gray)
        elif operation == "Boundary":
           imgout = c9.Boundary(img_gray)
        elif operation == "Contour":
           imgout = c9.Contour(img_gray)  # Tránh lỗi unpacking

        if imgout is not None:
           img_resized = cv2.resize(img, (300, 300))
           imgout_resized = cv2.resize(imgout, (300, 300))

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_resized, caption="Ảnh Gốc", use_column_width=True)
        with col2:
            st.image(imgout_resized, caption="Ảnh Xử Lý", use_column_width=True)

        return imgout
     
            
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png", "bmp", "tif", "webp"])
    
    if uploaded_file:
        data = uploaded_file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            st.error("⚠ Không thể tải ảnh. Hãy thử ảnh khác!")
        else:
            st.image(img, caption="📷 Ảnh đã tải lên", use_column_width=True)

            # Lựa chọn chương xử lý ảnh số
            chapter_choice = st.sidebar.radio("Chọn chương:", ["Chapter 3", "Chapter 4", "Chapter 9"])

            if chapter_choice == "Chapter 3":
                img = chapter3_operations(img)
            elif chapter_choice == "Chapter 4":
                img = chapter4_operations(img)
            elif chapter_choice == "Chapter 9":
                img = chapter9_operations(img)

            # Lưu ảnh đã xử lý
            save_image(img)

        


# === 8) Đọc chữ viết tay ===
elif menu == "Đọc chữ viết tay":
    st.header("✍️ Đọc chữ viết tay")
    uploaded = st.file_uploader("Tải ảnh viết tay lên", type=["jpg","jpeg","png","bmp"])
    if uploaded:
       data = uploaded.read()
       orig = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
       st.image(orig, caption="Ảnh viết tay gốc", use_column_width=True, clamp=True, channels="GRAY")

       x = preprocess_image_gray(orig)
       model_path = os.path.join("model", "best_model.h5")
       model_ocr = build_inference_model(model_path)
       preds = model_ocr.predict(x)
       seq = ctc_decode(preds)[0]
       text = num_to_text(seq)
       st.success(f"**Kết quả OCR:** {text}")

st.markdown("---")
st.caption("Demo bởi HieuDuong.")
