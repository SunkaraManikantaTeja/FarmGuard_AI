"""Model loading, prediction and Grad-CAM. Falls back to demo mode if no trained model is present."""
import os, json, io
import numpy as np
from PIL import Image

BASE = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH, LABELS_PATH = os.path.join(BASE, "model.keras"), os.path.join(BASE, "labels.json")
LOW_CONF = 0.60
_model = _labels = None

def available():
    return os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)

def _load():
    global _model, _labels
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        _labels = json.load(open(LABELS_PATH))
    return _model, _labels

def _prep(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
    return np.expand_dims(np.asarray(img, dtype="float32"), 0)  # model includes MobileNetV2 preprocessing

def predict(raw):
    model, labels = _load()
    probs = model.predict(_prep(raw), verbose=0)[0]
    top = probs.argsort()[::-1][:3]
    return {"prediction": labels[top[0]], "confidence": float(probs[top[0]]),
            "low_confidence": bool(probs[top[0]] < LOW_CONF),
            "alternatives": [{"label": labels[i], "confidence": float(probs[i])} for i in top[1:]]}

def gradcam(raw, last_conv="Conv_1"):
    """Returns a 224x224 heatmap (0..1) as list. Layer name may need changing for your model."""
    import tensorflow as tf
    model, _ = _load()
    x = _prep(raw)
    base = model.get_layer("mobilenetv2_1.00_224") if "mobilenetv2_1.00_224" in [l.name for l in model.layers] else model
    gm = tf.keras.models.Model(base.inputs, [base.get_layer(last_conv).output, base.output])
    with tf.GradientTape() as t:
        conv, out = gm(x); loss = tf.reduce_max(out, axis=-1)
    g = tf.reduce_mean(t.gradient(loss, conv), axis=(0, 1, 2))
    h = tf.nn.relu(tf.reduce_sum(conv[0] * g, axis=-1)).numpy()
    h = h / (h.max() + 1e-8)
    return np.asarray(Image.fromarray((h * 255).astype("uint8")).resize((224, 224))).astype("float32").div(255).tolist() if False else (np.asarray(Image.fromarray((h*255).astype("uint8")).resize((224,224)))/255.0).tolist()
