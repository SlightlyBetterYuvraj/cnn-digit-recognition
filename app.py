import os
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 10px;
    }
    .prediction-digit {
        font-size: 4.5rem;
        font-weight: 800;
        color: #60a5fa;
        line-height: 1;
        margin: 10px 0;
    }
    .confidence-text {
        font-size: 1.2rem;
        color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔢 Handwritten Digit Recognizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw any digit from 0 to 9 on the canvas below and let the CNN model predict it!</div>', unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cnn_model.joblib")

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Model file `cnn_model.joblib` not found. Please train and dump the model first by running `python train_and_dump.py`.")
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

col_canvas, col_prediction = st.columns([1, 1], gap="large")

with col_canvas:
    st.subheader("🎨 Drawing Canvas")
    st.caption("Draw your digit in white on the black board.")

    # Drawing tools control
    stroke_width = st.slider("Brush Stroke Width", min_value=10, max_value=35, value=20, step=1)

    # Canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        return_image_data=True,
    )

with col_prediction:
    st.subheader("🎯 Prediction & Analysis")

    if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 0):
        # Extract RGB/Grayscale from canvas
        raw_rgba = canvas_result.image_data.astype(np.uint8)
        img = Image.fromarray(raw_rgba).convert("L")

        # Bounding box crop and center to mimic MNIST preprocessing
        bbox = img.getbbox()
        if bbox is not None:
            # Crop to the drawn content
            cropped = img.crop(bbox)

            # Scale preserving aspect ratio into a 20x20 box (standard MNIST format)
            w, h = cropped.size
            if w > h:
                new_w = 20
                new_h = max(1, int(round((h * 20.0) / w)))
            else:
                new_h = 20
                new_w = max(1, int(round((w * 20.0) / h)))

            resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Paste centered on 28x28 black canvas
            processed_img = Image.new("L", (28, 28), color=0)
            paste_x = (28 - new_w) // 2
            paste_y = (28 - new_h) // 2
            processed_img.paste(resized, (paste_x, paste_y))
        else:
            processed_img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # Normalize identical to training xtrain / 255.0
        img_array = np.array(processed_img, dtype=np.float32) / 255.0
        input_tensor = img_array.reshape(1, 28, 28, 1)

        if model is not None:
            predictions = model.predict(input_tensor, verbose=0)[0]
            predicted_digit = int(np.argmax(predictions))
            confidence = float(predictions[predicted_digit]) * 100

            st.markdown(f"""
            <div class="prediction-box">
                <div class="confidence-text">Predicted Digit</div>
                <div class="prediction-digit">{predicted_digit}</div>
                <div class="confidence-text">Confidence: <strong>{confidence:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Class Probabilities")
            chart_data = {str(i): float(predictions[i]) for i in range(10)}
            st.bar_chart(chart_data)

            with st.expander("🔍 Processed 28x28 Input Preview"):
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.image(img, caption="Original Drawing (280x280)", width=140)
                with col_p2:
                    st.image(processed_img, caption="Normalized Input (28x28)", width=140)
    else:
        st.info("✏️ Draw a digit on the canvas to see the prediction.")
