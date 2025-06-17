import streamlit as st
import time
import cv2
import numpy as np
from ultralytics import YOLO
from email.message import EmailMessage
import smtplib, ssl
import os
from io import BytesIO
from PIL import Image
import base64

# -------------------------------
# 🔧 EMAIL ALERT FUNCTION
# -------------------------------
def send_email_alert(subject, body, to_email, image_bytes):
    sender_email = "akashsipcs@gmail.com"
    app_password = "zeyw jake wutq ppsd"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email
    msg.set_content(body)

    # Attach image
    msg.add_attachment(image_bytes, maintype="image", subtype="jpeg", filename="intrusion.jpg")

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, app_password)
        server.send_message(msg)

    st.success("📧 Email alert sent!")

# -------------------------------
# ✅ CONFIG & MODEL
# -------------------------------
st.set_page_config(page_title="Intrusion Alert", layout="wide")
st.title("🚨 Intrusion Detection System")

model = YOLO("yolov8n.pt")
names = model.names

allowed_labels = [
    'person', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'
]

# -------------------------------
# 📸 CAMERA INPUT
# -------------------------------
st.markdown("### 📷 Take a snapshot to scan for intrusions")
img_input = st.camera_input("Take a picture")

if img_input is not None:
    image_bytes = img_input.getvalue()
    np_img = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # -------------------------------
    # 🔍 Run Detection
    # -------------------------------
    results = model(frame)
    boxes = results[0].boxes
    detected_labels = set()
    person_detected = False

    for box in boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label = names[cls]

        if label in allowed_labels:
            detected_labels.add(label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if label == 'person':
                person_detected = True

    # -------------------------------
    # 📧 If person detected, send email
    # -------------------------------
    if person_detected:
        st.warning("🚨 Person detected! Sending email alert...")
        is_success, encoded_image = cv2.imencode('.jpg', frame)
        if is_success:
            send_email_alert(
                subject="🚨 Intruder Detected!",
                body="A person was detected in the camera snapshot.",
                to_email="receiver_email@gmail.com",
                image_bytes=encoded_image.tobytes()
            )

    # -------------------------------
    # 🔔 Play Chime (fallback for web)
    # -------------------------------
    if detected_labels and not person_detected:
        st.success("✅ Alert: Animal/Vehicle detected!")
        st.markdown("*Chime alert would play on local system.*")

    # -------------------------------
    # 🖼️ Show Output + Detected Labels
    # -------------------------------
    st.image(frame, channels="BGR", caption="Detected objects", use_column_width=True)

    with st.expander("📋 Detected Objects"):
        st.write(", ".join(detected_labels) if detected_labels else "No valid detections.")

    # -------------------------------
    # 💾 Download Image
    # -------------------------------
    st.download_button(
        label="📥 Download Detection Image",
        data=encoded_image.tobytes(),
        file_name=f"intrusion_{int(time.time())}.jpg",
        mime="image/jpeg"
    )

