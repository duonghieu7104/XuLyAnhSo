import streamlit as st
from face_recognition_utils import recognize_from_ip_camera
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from video_processor import VideoProcessor
from stream_processor import VProcessor
import cv2
import numpy as np
import tempfile

# Cấu hình TURN/STUN
RTC_CONFIGURATION = RTCConfiguration(
    {
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
)

st.title("Ứng dụng xử lý hình ảnh")

menu = st.sidebar.radio(
    "Chọn chức năng:",
    ("Nhận diện khuôn mặt", "Nhận diện đối tượng", "Xử lý ảnh")
)

def start_streaming():
    webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VProcessor,
        async_processing=True,
    )

if menu == "Nhận diện khuôn mặt":
    st.header("🔍 Nhận diện khuôn mặt")
    option = st.radio(
        "Chọn nguồn đầu vào:",
        ("Tải ảnh", "Tải video", "Chụp ảnh", "Stream Camera")
    )

    if option == "Tải ảnh":
        uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Tạo 2 cột để hiển thị ảnh ngang nhau
            col1, col2 = st.columns(2)

            # Hiển thị ảnh gốc ở cột đầu tiên với kích thước nhỏ hơn
            with col1:
                st.image(img, caption="Ảnh gốc", use_column_width=True, channels="BGR")

            processor = VideoProcessor()
            result_img, result_text = processor.process_frame(img)

            # Hiển thị ảnh kết quả nhận diện ở cột thứ hai với kích thước nhỏ hơn
            with col2:
                st.image(result_img, caption="Kết quả nhận diện", use_column_width=True, channels="BGR")

            # Hiển thị kết quả nhận diện dưới ảnh kết quả
            if result_text:
                st.text(f"Nhận diện: {result_text}")

        

    elif option == "Tải video":
        uploaded_video = st.file_uploader("Tải video lên", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            processor = VideoProcessor()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = processor.process_frame(frame)
                stframe.image(frame, channels="BGR")
            cap.release()
            

    elif option == "Chụp ảnh":
        st.write("Kết nối tới IP Camera...")
        ip_url = st.text_input("Nhập URL IP Camera", value="http://192.168.0.101:8080/video")

        if st.button("Nhận diện"):
            with st.spinner("Đang nhận diện..."):
                img, result_text = recognize_from_ip_camera(ip_url)
                if img is not None:
                    st.image(img, caption=f"Kết quả: {result_text}", channels="BGR")
                else:
                    st.error(result_text)

    elif option == "Stream Camera":
        st.subheader("📡 Streaming từ webcam (WebRTC)")
        start_streaming()
        

elif menu == "Nhận diện đối tượng":
    st.header("🧠 Nhận diện đối tượng")
    st.info("Chức năng đang được phát triển...")

elif menu == "Xử lý ảnh":
    st.header("🖼️ Xử lý ảnh")
    st.info("Chức năng đang được phát triển...")

st.markdown("---")
st.caption("Ứng dụng demo bởi HieuDuong.")
