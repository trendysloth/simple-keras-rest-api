# import the necessary packages
import tensorflow as tf
from keras import models
from PIL import Image
import numpy as np
import flask
import io
import base64
from flask_cors import CORS

# initialize Flask application
app = flask.Flask(__name__)
CORS(app)
model = None

def load_model():
	# load model
	global model
	model = models.load_model('./model.h5')

@app.route("/predict", methods=["POST"])
def predict():
    # Get posted image, convert to base64 string
    input_base64 = flask.request.get_json()['input_image']
    model_name = flask.request.get_json()['model']
    imgdata = base64.b64decode(input_base64.split(',')[1])
    image = Image.open(io.BytesIO(imgdata))
    image = image.convert("RGB")
    input_image_processed = tf.image.convert_image_dtype(np.array(image), tf.float32)
    input_image_processed = tf.image.resize(input_image_processed, (256, 256), antialias=True)
    input_image_processed = tf.expand_dims(input_image_processed, axis=[0]) # (1, 256, 256, 3)
    
    # predict output using model
    prediction = model(input_image_processed, training=True)
    output_image = prediction[0] # -> (256, 256, 3)
    array_from_tensor = np.asarray(output_image)
    # Remap the pixel values from float values to integer values
    mapped_array = np.interp(array_from_tensor, (0, +1), (0, +255))
    # Construct a PIL image to display in Runway
    PIL_IMG = Image.fromarray(np.uint8(mapped_array))
    b = io.BytesIO()
    PIL_IMG.save(b, 'jpeg')
    im_bytes = b.getvalue()
    encoded_string = base64.b64encode(im_bytes)
    return encoded_string

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		   "please wait until server has fully started"))
	load_model()
	app.run(debug=True)