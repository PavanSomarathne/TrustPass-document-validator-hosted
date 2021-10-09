import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

model = load_model('app/CNN_BR_Model_v2.h5')

app = Flask(__name__)
CORS(app)


@app.route('/validate', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        test_image = image.load_img(file_path, target_size=(375, 250))
        # Add a 3rd Color dimension to match Model expectation
        test_image = image.img_to_array(test_image)
        # Add one more dimension to beginning of image array so 'Predict' function can receive it (corresponds to Batch, even if only one batch)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        print(result)
        br = np.round(result[0][0], 2)
        cr = np.round(result[0][1], 2)
        uk = np.round(result[0][2], 2)

        message = "Image has low details"
        type = ""
        acc = 0
        out = False
        if(br > 0.5):
            message = "Acceptable BR"
            out = True
            type = "BR"
            acc = br
        if(cr > 0.5):
            out = True
            message = "Acceptable CR"
            type = "CR"
            acc = cr
        if((uk > 0.5 and br > 0.5) or (cr > 0.5 and br > 0.5) or (cr > 0.5 and uk > 0.5) or uk > 0.5):
            message = "Wrong Document"
            out = False

        if os.path.isfile(file_path):
            os.remove(file_path)
        print({'result': out, 'message': message,
               'type': type, 'accuracy': str(acc)})
        return jsonify(result=out, message=message, type=type, accuracy=str(acc))

    return None


@app.route("/", methods=['GET'])
def default():
    return "<h1> Sri Lanka ID Documents Indentifier <h1>"


if __name__ == "__main__":
    app.run()
