import streamlit as st
import os
import cv2
import numpy as np
import tempfile

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
    ("Nhận diện khuôn mặt", "Nhận diện đối tượng", "Xử lý ảnh"),
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


# === 7) Xử lý ảnh ===
elif menu == "Xử lý ảnh":
    st.header("🖼️ Xử lý ảnh")
    st.info("Chức năng đang được phát triển...")

st.markdown("---")
st.caption("Demo bởi HieuDuong.")
