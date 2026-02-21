from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import load_img, img_to_array # type: ignore
from werkzeug.utils import secure_filename
import numpy as np
import os
import uuid
import random
import traceback

# ================== FLASK APP ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# ================== PATHS ==================
MODEL_PATH = os.path.join(BASE_DIR, "model", "Vgg16-diabetes-best.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DATASET_FOLDER = os.path.join(BASE_DIR, "static", "Dataset", "testing")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================== ALLOWED FILES ==================
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ================== LOAD MODEL ==================
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model!")
    traceback.print_exc()
    model = None
    print("⚠️ Using dummy model for testing purposes.")

    class DummyModel:
        def predict(self, x):
            return np.array([[0.1, 0.2, 0.3, 0.15, 0.25]])
    model = DummyModel()

# ================== CLASS NAMES ==================
class_names = ["0", "1", "2", "3", "4"]

# ================== ROUTES ==================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if model is None:
            return "❌ Model not loaded. Check server logs."
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if not allowed_file(file.filename):
            return "❌ Only PNG, JPG, JPEG files are allowed."

        # Save uploaded file
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Preprocess image
        img = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        pred_class = class_names[np.argmax(preds[0])]
        confidence = np.max(preds[0]) * 100

        # Get related images
        class_folder = os.path.join(DATASET_FOLDER, pred_class)
        related_imgs_paths = []

        if os.path.exists(class_folder):
            imgs = [img for img in os.listdir(class_folder) if allowed_file(img)]
            if imgs:
                random.shuffle(imgs)
                selected_imgs = imgs[:5]
                related_imgs_paths = [f"testing/{pred_class}/{img}" for img in selected_imgs]

        return render_template(
            "predict.html",
            filename=f"uploads/{filename}",
            prediction=pred_class,
            confidence=round(confidence, 2),
            related_images=related_imgs_paths
        )

    return render_template("predict.html")

@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename=f"uploads/{filename}"), code=301)

# ================== LOGIN & REGISTER ROUTES ==================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        # Simple placeholder authentication
        if username == "admin" and password == "admin":
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        # Placeholder: Save user info to DB later
        return redirect(url_for("login"))
    return render_template("register.html")

# ================== RUN APP ==================
if __name__ == "__main__":
    app.run(debug=True)

