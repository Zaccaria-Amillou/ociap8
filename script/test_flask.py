import os
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf
from flask import Flask, request, send_file, render_template
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras import backend as K


NB_IMAGES = 20
selected_id = 1
img_paths = 'static/data/img/'
mask_paths = 'static/data/mask/'

cats = {'void': [0, 1, 2, 3, 4, 5, 6],
 'flat': [7, 8, 9, 10],
 'construction': [11, 12, 13, 14, 15, 16],
 'object': [17, 18, 19, 20],
 'nature': [21, 22],
 'sky': [23],
 'human': [24, 25],
 'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]
 }

cats_id = {
 'void': (0),
 'flat': (1),
 'construction': (2),
 'object': (3),
 'nature': (4),
 'sky': (5),
 'human':(6),
 'vehicle': (7)
}

cats_colors = {
 0: (0,0,0),
 1: (50,50,50),
 2: (150,150,150),
 3: (255,0,0),
 4: (0,255,0),
 5: (0,0,255),
 6:(200,200,0),
 7: (150,0,200)
}

# Fonctions loss
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.cast(K.flatten(y_true), K.floatx())
    y_pred_f = K.cast(K.flatten(y_pred), K.floatx())
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss

# Prépare les données pour la segmentation avec model.predict()
def get_data_prepared(path_X, dim):
    X = np.array([cv2.resize(cv2.cvtColor(cv2.imread(path_X), cv2.COLOR_BGR2RGB), dim)])
    X = X / 255

    return X

# Prépare l'image pour la segmentation
def prepare_img(img, dim):
    X = np.array([cv2.resize(np.array(img), dim)])
    X = X / 255

    return X

# Recupère les chemins d'accès des fichiers
def getPathFiles():
    path_files = []

    # img set
    for file in os.listdir(img_paths):
        path_files.append(file.replace('leftImg8bit.png',''))

    return path_files


path_files = getPathFiles()

app = Flask(__name__)

model = tf.lite.Interpreter(model_path='../model/checkpointVGG19_U-Net_base.tflite')
model.allocate_tensors()

# Endpoint pour récupérer l'image et le masque réels en fonction de l'id sélectionné
@app.route('/', methods=["GET","POST"])
def index():

    global selected_id
    if request.form.get('file'):
        selected_id = int(request.form.get('file'))

    img_path = img_paths + path_files[selected_id-1] + 'leftImg8bit.png'
    mask_path = mask_paths + path_files[selected_id-1] + 'gtFine_labelIds.png'

    img = cv2.resize(cv2.imread(img_path), (400, 200))
    mask = cv2.resize(cv2.imread(mask_path), (400, 200))
    mask = np.squeeze(mask[:,:,0])
    mask_labelids = np.zeros((mask.shape[0], mask.shape[1], len(cats_id)))

    for i in range(-1, 34):
        for cat in cats:
            if i in cats[cat]:
                mask_labelids[:,:,cats_id[cat]] = np.logical_or(mask_labelids[:,:,cats_id[cat]],(mask==i))
                break

    mask_labelids = np.array(np.argmax(mask_labelids,axis=2), dtype='uint8')

    m = np.empty((mask_labelids.shape[0], mask_labelids.shape[1], 3), dtype='uint8')
    for i in range(mask_labelids.shape[0]):
        for j in range(mask_labelids.shape[1]):
            m[i][j] = cats_colors[mask_labelids[i][j]]

    cv2.imwrite('static/data/predict/img.png', img)
    cv2.imwrite('static/data/predict/mask.png', m)

    return render_template('index.html', sended=False, nb_image=NB_IMAGES, selected=selected_id)
    

# Endpoint pour segmenter l'image choisie à partir de l'app flask
@app.route('/predict/', methods=["POST"])
def predictImage():
    img_path = img_paths + path_files[selected_id-1] + 'leftImg8bit.png'

    img = get_data_prepared(img_path, (256,256))
    img = img.astype('float32')  # Convert the input tensor to float32
    input_details = model.get_input_details()
    model.set_tensor(input_details[0]['index'], img)
    model.invoke()
    y_pred = model.get_tensor(model.get_output_details()[0]['index'])
    y_pred_argmax=np.argmax(y_pred, axis=3)

    m = np.empty((y_pred_argmax[0].shape[0],y_pred_argmax[0].shape[1],3), dtype='uint8')
    for i in range(y_pred_argmax[0].shape[0]):
        for j in range(y_pred_argmax[0].shape[1]):
            m[i][j] = cats_colors[y_pred_argmax[0][i][j]]

    cv2.imwrite('static/data/predict/mask_predicted.png', cv2.resize(m, (400,200)))

    background = cv2.imread('static/data/predict/img.png')
    overlay = cv2.imread('static/data/predict/mask_predicted.png')

    added_image = cv2.addWeighted(background,1,overlay,0.6,0)

    cv2.imwrite('static/data/predict/combined.png', added_image)

    return render_template('index.html',sended=True,nb_image=NB_IMAGES, selected=selected_id)

# Endpoint pour segmenter une image quelconque envoyée depuis une requête API
@app.route('/segment/', methods=["POST"])
def segmentImage():

    test = request.files.get('image')

    if test:
        image = Image.open(test)

        img = prepare_img(image, (256,256))
        y_pred = model.predict(img)
        y_pred_argmax=np.argmax(y_pred, axis=3)

        m = np.empty((y_pred_argmax[0].shape[0],y_pred_argmax[0].shape[1],3), dtype='uint8')
        for i in range(y_pred_argmax[0].shape[0]):
            for j in range(y_pred_argmax[0].shape[1]):
                m[i][j] = cats_colors[y_pred_argmax[0][i][j]]

        m = cv2.resize(m, (400,200))

        im_bgr = cv2.cvtColor(m, cv2.COLOR_RGB2BGR)

        img = Image.fromarray(im_bgr)

        file_object = io.BytesIO()

        img.save(file_object, 'png')

        file_object.seek(0)

        return send_file(file_object, mimetype='image/png')
    
    else:
        return "Image not send", 400

if __name__ == "__main__":
    app.run()