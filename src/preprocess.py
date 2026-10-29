"""Turn a hand-drawn canvas image into an MNIST-style 28x28 input vector."""
import numpy as np
from PIL import Image


def preprocess_canvas(img, size=28, ink_threshold=20):
    """img: HxW, HxWx3 or HxWx4 array, white ink on a black background.

    MNIST digits are cropped, scaled to fit a 20x20 box (aspect ratio kept) and
    centred by their centre of mass inside a 28x28 frame. Doing the same here
    makes drawn digits much closer to the training distribution.
    Returns (vector of length size*size scaled to [0, 1], 28x28 array) or None if blank.
    """
    arr = np.asarray(img)
    gray = arr[..., :3].astype(np.float64).mean(axis=2) if arr.ndim == 3 else arr.astype(np.float64)
    ys, xs = np.where(gray > ink_threshold)
    if len(ys) == 0:
        return None

    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = np.asarray(
        Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS),
        dtype=np.float64,
    )

    total = small.sum()
    cy = (small.sum(axis=1) * np.arange(new_h)).sum() / total
    cx = (small.sum(axis=0) * np.arange(new_w)).sum() / total
    top = int(np.clip(round(size / 2 - cy), 0, size - new_h))
    left = int(np.clip(round(size / 2 - cx), 0, size - new_w))

    frame = np.zeros((size, size))
    frame[top:top + new_h, left:left + new_w] = small
    frame = frame / 255.0
    return frame.reshape(-1).astype(np.float32), frame
