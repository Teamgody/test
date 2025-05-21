
from flask import Flask, request, render_template
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = MobileNetV2(weights="imagenet")

def classify_image_ai(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    preds = model.predict(image_array)
    decoded = decode_predictions(preds, top=1)[0][0]
    return f"{decoded[1]} ({decoded[2]*100:.2f}%)"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join("static", file.filename)
            file.save(image_path)
            result = classify_image_ai(image_path)
    return render_template("index.html", result=result, image=image_path)

if __name__ == "__main__":
    app.run(debug=True)
