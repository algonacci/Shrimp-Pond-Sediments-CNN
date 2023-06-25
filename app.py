import datetime
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
np.set_printoptions(suppress=True)


app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_FILE'] = 'keras_model.h5'
app.config['LABELS_FILE'] = 'labels.txt'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


model = load_model(app.config['MODEL_FILE'], compile=False)
with open(app.config['LABELS_FILE'], 'r') as file:
    labels = file.read().splitlines()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        image = request.files["image"]
        # tanggal = request.form["tanggal"]
        usia = int(request.form["usia"])
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            # processed_image = md.image_processing(image_path)
            img = Image.open(image_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.asarray(img)
            img_array = np.expand_dims(img_array, axis=0)
            normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            start_time = time.time()
            predictions = model.predict(data)
            index = np.argmax(predictions)
            class_name = labels[index]
            confidence_score = predictions[0][index]
            print("===================================")
            print("Class:", class_name[2:], end="")
            print("\n")
            print("Confidence Score:", confidence_score)
            print("===================================")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed Time:", elapsed_time, "seconds")

            # Calculate shift_pond based on usia and predicted class
            # year, month, day = map(int, tanggal.split('-'))
            # input_date = datetime.date(year, month, day)

            if usia <= 30:
                shift_pond = 30 - usia
                shift_date = datetime.date.today() + datetime.timedelta(days=shift_pond)
            elif np.argmax(predictions) == 0:
                shift_pond = 4
                shift_date = datetime.date.today() + datetime.timedelta(days=shift_pond)
            elif np.argmax(predictions) == 1:
                shift_pond = 3
                shift_date = datetime.date.today() + datetime.timedelta(days=shift_pond)
            elif np.argmax(predictions) == 2:
                shift_pond = 2
                shift_date = datetime.date.today() + datetime.timedelta(days=shift_pond)
            else:
                shift_pond = None
                shift_date = None

            return render_template("prediction.html",
                                   result=class_name,
                                   probabilities=confidence_score,
                                   shift_pond=shift_pond,
                                   shift_date=shift_date,
                                   prediction_result=image_path,
                                   usia=usia)
        else:
            return render_template("prediction.html", "result.html", input_date=input_date, error="Silahkan upload gambar dengan format JPG")
    else:
        return render_template("prediction.html")


if __name__ == "__main__":
    app.run()
