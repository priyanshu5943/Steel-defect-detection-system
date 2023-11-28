from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import io
from werkzeug.utils import secure_filename
import pandas as pd
import os
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import keras
from keras.layers import UpSampling2D, Conv2D, Activation, Input, Dropout, MaxPooling2D
from keras import Model
from keras import backend as K


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Define the UPLOAD_FOLDER where uploaded images will be stored
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained segmentation model
model = load_model('steel_model.h5')

casting_model = load_model('casting.hdf5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



img_size = 256

def process_image(filename):
    # Load the uploaded image and perform preprocessing
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    original_img = cv2.imread(img_path)

    # Resize the image to the desired dimensions
    original_img = cv2.resize(original_img, (256, 256))

    # Perform the prediction on the specified image.
    predict = model.predict(np.asarray([original_img]))

    # Reshape and process the prediction to generate the mask.
    predicted_img = cv2.resize(predict[0], (1600, img_size))
    tmp = np.copy(predicted_img)
    tmp[tmp < np.mean(predicted_img)] = 0
    tmp[tmp > 0] = 1

    # Save the segmented image
    segmented_image_filename = os.path.join(app.config['UPLOAD_FOLDER'], "segmented_" + filename)
    plt.imsave(segmented_image_filename, tmp)

    return segmented_image_filename
def preprocess_image(uploaded_file):
    img_io = io.BytesIO(uploaded_file.read())
    img = image.load_img(img_io, target_size=(299, 299), color_mode="rgb") # Adjusted target_size and color_mode
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_defect(img_array):
    predictions = casting_model.predict(img_array)
    if predictions[0][0] > 0.5:
        return "ok-front"
    else:
        return "def-front"

    

# Load the steel strength prediction model
with open("model_vr.pkl", "rb") as f:
    steel_model = pickle.load(f)

# Load the concrete strength prediction model
with open("modeli.pkl", "rb") as g:
    concrete_model = pickle.load(g)

# Load the epoxy viscosity prediction model
epoxy_model = pickle.load(open("modelepox.pkl", "rb"))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def welcome():
    return render_template("welcome.html")

@app.route('/ml')
def ml_page():
    return render_template("ml.html")

@app.route('/main')
def main_page():
    return render_template("main.html")

@app.route("/epoxy")
def epoxy_home():
    return render_template("epoxy.html")

@app.route("/epoxy_predict", methods=["GET", "POST"])
def epoxy_predict():
    if request.method == "POST":
        try:
            # Numeric features
            temperature = float(request.form['temperature'])
            ratio = float(request.form['ratio_of_epoxy_resin_to_diluent'])

            # Categories for which one-hot encoding is needed
            categories = {
                "epoxy_resin": ["bisphenol_a", "ester_cyclic", "phenolic"],
                "additional_group_of_epoxy_resin": ["amyl_phenol_acid_", "h", "acrylate", "acrylate_", "methacrylate"],
                "diluent": ["furan_acetone", "others", "acrylate", "epoxy_oil", "ester", "glycidyl_ether", "phenol", "phenolic", "polysiloxane", "siloxane_"],
                "additional_group_of_diluent": [
                    "1_4butanediol", "1_6hexanediol", "bisphosphonatepiperazine_hydroxyl",
                    "cardanol_was_grafted_with_silicone", "dipropylene_glycol", "epoxy_ricin_acid", "h",
                    "parsley_methyl_group", "polyoxycardanol", "anacardol_", "benzyl_alcohol", "butyl_",
                    "cyclohexene_oxide", "diglycol", "epoxy_group", "eugenol", "heterocycle", "methylene",
                    "methyl_", "nbutyl", "nonyl", "oxylene", "octyl_group_", "paminophenol", "phenyl",
                    "phenylethylene_oxide", "phosphoric_acid", "propylene_glycol", "styrene_", "thymol"
                ],
                # Add more categories here...
            }

            input_features = [temperature, ratio]

            for feature, possible_values in categories.items():
                category = request.form[feature]
                one_hot = [1 if val == category else 0 for val in possible_values]
                input_features += one_hot

            prediction = epoxy_model.predict([input_features])

            return render_template('epoxy.html', prediction_text="Predicted Epoxy Viscosity is {:.2f} Centipoise".format(prediction[0]))

        except Exception as e:
            return render_template('epoxy.html', prediction_text="Error: {}".format(str(e)))

    return render_template("epoxy.html")

@app.route('/predict', methods=["GET", "POST"])
def predict_steel():
    prediction = None
    if request.method == "POST":
        features = [request.form[feature] for feature in ['c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']]
        prediction = steel_model.predict([features])[0]

        # Save the prediction to session
        session['prediction'] = prediction

        # Redirect to the result page for steel
        return redirect(url_for('result_steel'))

    return render_template("indexi.html")

@app.route('/result')
def result_steel():
    prediction = session.get('prediction')
    return render_template("result.html", prediction=prediction)

@app.route('/casting', methods=["GET", "POST"])
def casting_page():
    return render_template('casting.html')

@app.route('/casting_predict', methods=["POST"])
def casting_predict():
    if 'file' not in request.files:
        return render_template('casting.html', error="No file part in the request.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('casting.html', error="No file selected.")

    if file and allowed_file(file.filename):
        try:
            img_array = preprocess_image(file)
            prediction = predict_defect(img_array)
            
            image_path = "static/uploaded_image.jpg"
            file.seek(0)
            file.save(image_path)
            
            return render_template('casting.html', prediction_text=f'The prediction is: {prediction}', image_path=image_path)
        except Exception as e:
            return render_template('casting.html', error=f"Error processing the image: {str(e)}")
    
    return render_template('casting.html', error="Invalid file type.")

@app.route('/steel_defect')
def home():
    return render_template('stl.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Process the uploaded image and get the segmented image and mask
        segmented_image = process_image(filename)

       
        return render_template('steel.html', original_image=filename, segmented_image=os.path.basename(segmented_image))
    



@app.route('/concrete', methods=["GET", "POST"])
def predict_concrete():
    prediction = None
    if request.method == "POST":
        features = [request.form[feature] for feature in ['CementComponent', 'BlastFurnaceSlag', 'FlyAshComponent', 'WaterComponent', 'FineAggregateComponent', 'AgeInDays']]
        prediction = concrete_model.predict([features])[0]

        # Save the prediction to session
        session['concrete_prediction'] = prediction

        # Redirect to the result page for concrete
        return redirect(url_for('result_concrete'))

    return render_template("concrete.html")

@app.route('/concrete_result')
def result_concrete():
    prediction = session.get('concrete_prediction')
    return render_template("concrete_result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)

