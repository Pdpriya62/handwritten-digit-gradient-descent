"""Streamlit demo: draw a digit, get a prediction from the Gradient-Descent-trained model.

    python run_all.py         # once, to train and save results/weights.npz
    streamlit run app.py
"""
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src.model import SoftmaxClassifier
from src.preprocess import preprocess_canvas

WEIGHTS = Path(__file__).resolve().parent / "results" / "weights.npz"

st.set_page_config(page_title="Digit classifier", page_icon="🔢")
st.title("Handwritten digit classifier")
st.caption("Softmax Regression trained with Gradient Descent on MNIST (NumPy from scratch)")


@st.cache_resource
def load_model():
    return SoftmaxClassifier.load(WEIGHTS)


if not WEIGHTS.exists():
    st.error("No trained weights found. Run `python run_all.py` first, then reload this page.")
    st.stop()

model = load_model()
if model.W.shape[1] != 784:
    st.error("The saved weights are not for 28x28 MNIST images. Re-run `python run_all.py` (default MNIST).")
    st.stop()

left, right = st.columns([1, 1])
with left:
    st.write("Draw one digit, large and centred:")
    canvas = st_canvas(
        fill_color="#000000", stroke_width=20, stroke_color="#FFFFFF",
        background_color="#000000", height=280, width=280,
        drawing_mode="freedraw", key="canvas",
    )
    st.caption("Use the trash icon under the canvas to clear it.")

with right:
    result = None
    if canvas.image_data is not None:
        result = preprocess_canvas(canvas.image_data)
    if result is None:
        st.info("Draw a digit to see the prediction.")
    else:
        vec, frame = result
        probs = model.predict_proba(vec[None, :])[0]
        pred = int(probs.argmax())
        st.metric("Predicted digit", pred, f"{probs[pred] * 100:.1f}% confidence")
        st.write("What the model sees (28x28):")
        st.image(frame, width=140, clamp=True)
        st.bar_chart({"probability": {str(k): float(probs[k]) for k in range(10)}})
