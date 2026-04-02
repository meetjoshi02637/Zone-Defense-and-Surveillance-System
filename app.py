import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import time
import datetime
import pandas as pd

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    try:
        general_model = YOLO("yolov8n.pt")
        custom_model = YOLO(os.path.join(os.getcwd(), "best.pt"))
        return general_model, custom_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None

general_model, custom_model = load_models()

if general_model is None:
    st.stop()

# ---------------- SESSION LOG ----------------
if "logs" not in st.session_state:
    st.session_state.logs = []

# ---------------- ZONE DETECTION ----------------
def check_zone_intrusion(results, model, frame_shape):
    intrusion = False
    h, w = frame_shape[:2]

    zone_x1, zone_y1 = int(w * 0.3), int(h * 0.3)
    zone_x2, zone_y2 = int(w * 0.7), int(h * 0.7)

    for box in results[0].boxes or []:
        cls = int(box.cls[0].item())
        label = model.names[cls].lower()

        if label == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if (x1 < zone_x2 and x2 > zone_x1 and
                y1 < zone_y2 and y2 > zone_y1):
                intrusion = True

    return intrusion, (zone_x1, zone_y1, zone_x2, zone_y2)

# ---------------- THREAT ENGINE ----------------
def get_smart_threat(img, res_gen, res_custom, model):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if brightness < 70:
        mode = "NIGHT"
        score = 3
    elif brightness < 120:
        mode = "DIM"
        score = 2
    else:
        mode = "DAY"
        score = 1

    person_count = 0
    vehicle_count = 0

    vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]

    if res_gen[0].boxes is not None:
        for box in res_gen[0].boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = model.names[cls].lower()

            if conf < 0.25:
                continue

            if label == "person":
                person_count += 1
            elif label in vehicle_classes:
                vehicle_count += 1

    custom_detected = 0
    if res_custom[0].boxes is not None:
        for box in res_custom[0].boxes:
            if float(box.conf[0].item()) > 0.3:
                custom_detected += 1

    # -------- LOGIC --------
    if person_count > 0:
        score += 3

    if vehicle_count > 0:
        score += 1

    if mode == "NIGHT":
        score += 2

    if mode == "NIGHT" and person_count > 0:
        score += 4

    if custom_detected > 0:
        score += 5

    # -------- FINAL --------
    if score <= 2:
        threat = "🟢 LOW"
    elif score <= 5:
        threat = "🟡 MEDIUM"
    elif score <= 9:
        threat = "🟠 HIGH"
    else:
        threat = "🔴 CRITICAL"

    return threat, mode, brightness, person_count, vehicle_count, custom_detected

# ---------------- UI ----------------
st.set_page_config(page_title="AI Surveillance", layout="wide")

st.title("🚨 AI Border Surveillance System")

confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.25)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ---------------- IMAGE MODE ----------------
if uploaded_file:
    image = Image.open(uploaded_file)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, use_container_width=True)

    # -------- DETECTION --------
    res_gen = general_model(img, conf=confidence)
    res_custom = custom_model(img, conf=confidence)

    annotated = res_gen[0].plot()

    # -------- ZONE --------
    intrusion, zone = check_zone_intrusion(res_gen, general_model, img.shape)
    x1, y1, x2, y2 = zone
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

    st.image(annotated, use_container_width=True)

    # -------- THREAT --------
    threat, mode, brightness, p, v, c = get_smart_threat(
        img, res_gen, res_custom, general_model
    )

    st.subheader("🚨 Threat Level")
    st.metric("Threat", threat)

    if intrusion:
        st.error("🚨 INTRUSION DETECTED!")

    if threat == "🔴 CRITICAL":
        st.error("🚨 CRITICAL ALERT!")

    # -------- INFO --------
    st.subheader("🧠 Intelligence Info")
    st.write(f"🌙 Mode: {mode}")
    st.write(f"💡 Brightness: {brightness:.2f}")
    st.write(f"👤 Persons: {p}")
    st.write(f"🚗 Vehicles: {v}")
    st.write(f"🎯 Custom Threats: {c}")

    # -------- LOGGING --------
    log = {
        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
        "Mode": mode,
        "Threat": threat,
        "Persons": p,
        "Vehicles": v,
        "Custom": c,
        "Intrusion": intrusion
    }

    st.session_state.logs.append(log)

# ---------------- DASHBOARD ----------------
st.subheader("📊 Threat Logs Dashboard")

if st.session_state.logs:
    df = pd.DataFrame(st.session_state.logs[::-1])
    st.dataframe(df, use_container_width=True)

# ---------------- WEBCAM ----------------
run = st.sidebar.checkbox("Start Webcam")

FRAME = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res_gen = general_model(frame, conf=confidence)
        res_custom = custom_model(frame, conf=confidence)

        annotated = res_gen[0].plot()

        intrusion, zone = check_zone_intrusion(res_gen, general_model, frame.shape)
        x1, y1, x2, y2 = zone
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        threat, mode, brightness, p, v, c = get_smart_threat(
            frame, res_gen, res_custom, general_model
        )

        FRAME.image(annotated, channels="BGR", use_container_width=True)

        st.sidebar.metric("Threat", threat)

        time.sleep(0.03)

        if not st.session_state.get("Start Webcam", False):
            break

    cap.release()
