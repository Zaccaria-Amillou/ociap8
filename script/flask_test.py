from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='path_to_your_model.tflite')
interpreter.allocate_tensors()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file'].read()
    image = Image.open(io.BytesIO(file))
    image = np.array(image)

    # Make sure the image is in the correct format
    if image.shape != (desired_height, desired_width, num_channels):
        return jsonify({'error': 'invalid image dimensions'}), 400

    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)

    # Get the input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Predict the image
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Return the prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)