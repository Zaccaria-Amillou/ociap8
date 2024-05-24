import os
import pandas as pd
import numpy as np
import time
import datetime
import cv2
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.metrics import MeanIoU, OneHotMeanIoU
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras import backend as K

import pickle
from keras.models import load_model

CHECKPOINT_PATH = "../model/checkpoint/"
MODELPATH = "../model/"
HISTORY_PATH = "../model/history/"

monitor_val = "val_mean_iou"
learning_rate = 0.00001
nbr_epochs = 20
patience = 5

# Définir dimensions images
DIM_X = 256
DIM_Y = 256
INPUT_SHAPE = (DIM_X, DIM_Y, 3)
NB_CLASSE = 8

cats = {
    "void": [0, 1, 2, 3, 4, 5, 6],
    "flat": [7, 8, 9, 10],
    "construction": [11, 12, 13, 14, 15, 16],
    "object": [17, 18, 19, 20],
    "nature": [21, 22],
    "sky": [23],
    "human": [24, 25],
    "vehicle": [26, 27, 28, 29, 30, 31, 32, 33, -1],
}

cats_id = {
    "void": (0),
    "flat": (1),
    "construction": (2),
    "object": (3),
    "nature": (4),
    "sky": (5),
    "human": (6),
    "vehicle": (7),
}

cats_color = {
    "void": (0, 0, 0),  # Black
    "flat": (128, 0, 128),  # Purple
    "construction": (128, 128, 128),  # Grey
    "object": (255, 255, 0),  # Yellow
    "nature": (0, 128, 0),  # Green
    "sky": (173, 216, 230),  # Light Blue
    "human": (255, 0, 0),  # Red
    "vehicle": (0, 0, 255),  # Blue
}


# Converti le masque pour le réduire à 8 catégories (one hote encoding)
def convert_mask(img):
    """
    Convertit un masque d'image donné en un format d'encodage one-hot réduit à 8 catégories.

    Paramètres:
    img (numpy.ndarray): Le masque d'image d'entrée à convertir. On suppose qu'il s'agit d'un tableau numpy 2D ou 3D.

    Retourne:
    numpy.ndarray: Le masque d'image converti en format d'encodage one-hot. La forme du tableau retourné est (img.shape[0], img.shape[1], len(cats_id)).

    Remarque:
    La fonction utilise deux variables globales - cats et cats_id.
    cats est un dictionnaire où les clés sont les noms de catégories et les valeurs sont des listes d'entiers représentant les catégories originales dans l'image d'entrée qui correspondent à la nouvelle catégorie.
    cats_id est un dictionnaire où les clés sont les noms de catégories et les valeurs sont les nouveaux identifiants de catégorie (entiers de 0 à 7).
    """
    img = np.squeeze(img)
    mask = np.zeros((img.shape[0], img.shape[1], len(cats_id)), dtype="uint8")

    for i in range(-1, 34):
        for cat in cats:
            if i in cats[cat]:
                mask[:, :, cats_id[cat]] = np.logical_or(
                    mask[:, :, cats_id[cat]], (img == i)
                )
                break

    return np.array(mask, dtype="uint8")



def augment_data(X, y):
    seq = iaa.Sequential(
        [
            iaa.Fliplr(1.0),  # always flip every image
            #iaa.Affine(scale=(1.15)),  # scale images to 80-120% of their size
            #iaa.AdditiveGaussianNoise(scale=(10, 20)),  # add gaussian noise to images
            iaa.Multiply((1.2, 2.5)),  # multiply pixel values by 0.8-1.2 (randomly picked per image)
            #iaa.LinearContrast((0.50, 1.50)),  # scale contrast by 0.75-1.25 (randomly picked per image)
            #iaa.MultiplySaturation((0.5, 1.2))  # multiply saturation by 0.8-1.2 (randomly picked per image)
        ],
        random_order=False,  # apply augmentations in the same order every time
    )

    new_X = []
    new_y = []

    for i in range(len(X)):
        img = X[i]
        mask = y[i]
        segmap = SegmentationMapsOnImage(mask, shape=img.shape)

        img_aug, segmap_aug = seq(image=img, segmentation_maps=segmap)
        
        # Resize the augmented images and masks to the original image size
        img_aug = cv2.resize(img_aug, (img.shape[1], img.shape[0]))
        segmap_aug = cv2.resize(segmap_aug.get_arr(), (img.shape[1], img.shape[0]))

        new_X.append(img_aug)
        new_y.append(segmap_aug)

    new_X = np.array(new_X)
    new_y = np.array(new_y)

    return new_X, new_y

# Prépare les données pour la segmentation
def get_data_prepared(path_X_list, path_Y_list, dim):
    """
    Prépare les données pour la segmentation en lisant les images et les masques à partir des chemins donnés, en les redimensionnant à une dimension spécifiée, en convertissant les masques en un format d'encodage one-hot et en normalisant les images.

    Paramètres:
    path_X_list (list): Une liste de chemins vers les images à lire.
    path_Y_list (list): Une liste de chemins vers les masques d'image correspondants à lire.
    dim (tuple): Un tuple spécifiant la dimension à laquelle redimensionner les images et les masques.

    Retourne:
    tuple: Un tuple contenant deux tableaux numpy. Le premier tableau contient les images redimensionnées et normalisées, et le deuxième tableau contient les masques d'image convertis et redimensionnés.

    Note:
    La fonction utilise la fonction convert_mask pour convertir les masques d'image en un format d'encodage one-hot. Les images sont normalisées en divisant chaque pixel par 255.
    """
    X = np.array([cv2.resize(cv2.imread(path_X), dim) for path_X in path_X_list])
    y = np.array(
        [cv2.resize(convert_mask(cv2.imread(path_y, 0)), dim) for path_y in path_Y_list]
    )

    if len(y.shape) == 3:
        y = np.expand_dims(y, axis=3)
    X = X / 255

    return X, y


# Affiche graphiquement l'évolution de la fonction loss et de la métrique lors de l'entraînement d'un modèle
def draw_history(history):
    """
    Affiche graphiquement l'évolution de la fonction de perte (loss) et de la métrique (mean_iou) lors de l'entraînement d'un modèle.

    Paramètres:
    history (dict): Un dictionnaire contenant les historiques de la fonction de perte et de la métrique pour les ensembles d'entraînement et de validation. Les clés attendues sont "loss", "val_loss", "mean_iou" et "val_mean_iou".

    Cette fonction génère deux graphiques : le premier montre l'évolution de la fonction de perte pour les ensembles d'entraînement et de validation au fil des époques, et le deuxième montre l'évolution de la métrique mean_iou pour les ensembles d'entraînement et de validation au fil des époques.
    """
    plt.subplots(1, 2, figsize=(15, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(history["mean_iou"])
    plt.plot(history["val_mean_iou"])
    plt.title("model mean_iou")
    plt.ylabel("mean_iou")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")

    plt.show()


def getPathFiles(dir_path, pattern=""):
    """
    Récupère les chemins des fichiers correspondant à un certain motif dans les répertoires d'entraînement, de test et de validation.

    Paramètres:
    dir_path (str): Le chemin du répertoire contenant les sous-répertoires "train", "test" et "val".
    pattern (str): Le motif à rechercher dans les noms de fichiers. Seuls les fichiers dont le nom contient ce motif seront renvoyés. Par défaut, tous les fichiers sont renvoyés.

    Retourne:
    tuple: Un tuple contenant trois listes de chemins de fichiers. La première liste contient les chemins des fichiers d'entraînement, la deuxième liste contient les chemins des fichiers de test et la troisième liste contient les chemins des fichiers de validation.

    Note:
    La fonction ignore les fichiers ".DS_Store" qui sont créés par le système d'exploitation macOS.
    """
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    val_path = dir_path + "val/"

    train_path_files = []
    test_path_files = []
    val_path_files = []

    # Train set
    for dir in os.listdir(train_path):
        if dir == ".DS_Store":
            continue
        for file in os.listdir(train_path + dir):
            if file == ".DS_Store":
                continue
            if pattern in file:
                train_path_files.append(train_path + dir + "/" + file)

    # Test set
    for dir in os.listdir(test_path):
        if dir == ".DS_Store":
            continue
        for file in os.listdir(test_path + dir):
            if file == ".DS_Store":
                continue
            if pattern in file:
                test_path_files.append(test_path + dir + "/" + file)

    # Val set
    for dir in os.listdir(val_path):
        if dir == ".DS_Store":
            continue
        for file in os.listdir(val_path + dir):
            if file == ".DS_Store":
                continue
            if pattern in file:
                val_path_files.append(val_path + dir + "/" + file)

    return train_path_files, test_path_files, val_path_files

def load_model_histories(history_directory, time_directory):
    """
    Load model histories and training times from .pkl files in two directories.

    Parameters:
    history_directory (str): The directory containing the history .pkl files.
    time_directory (str): The directory containing the training time .pkl files.

    Returns:
    pd.DataFrame: A DataFrame containing the model histories and training times.
    """

    # List all the .pkl files in the directories
    history_files = [f for f in os.listdir(history_directory) if f.endswith('_history.pkl')]
    time_files = [f for f in os.listdir(time_directory) if f.endswith('train_times.pkl')]

    # Initialize a list to store DataFrames
    dfs = []

    # Loop through the .pkl files
    for history_file in history_files:
        # Extract the model name from the file name
        model_name = history_file.replace('_history.pkl', '')
        
        # Load the history from the .pkl file
        with open(os.path.join(history_directory, history_file), 'rb') as f:
            history = pickle.load(f)
        
        # Convert the history into a DataFrame
        history_df = pd.DataFrame(history)
        
        # Add a column for the model name
        history_df['model_name'] = model_name

        # Find the corresponding time file
        time_file = model_name + 'train_times.pkl'
        if time_file in time_files:
            # Load the training times from the .pkl file
            with open(os.path.join(time_directory, time_file), 'rb') as f:
                train_times_dict = pickle.load(f)
            
            # Extract the training time from the dictionary
            train_time_seconds = train_times_dict.get(model_name, None)
            
            # Convert the training time to a more readable format (HhMmSs)
            if train_time_seconds is not None:
                dt = datetime.timedelta(seconds=int(train_time_seconds))
                train_time = f"{dt.days * 24 + dt.seconds // 3600}h {dt.seconds // 60 % 60}m {dt.seconds % 60}s"
            else:
                train_time = None
            
            # Add the training time to the DataFrame
            history_df['train_time'] = train_time

        # Add the history DataFrame to the list
        dfs.append(history_df)

    # Concatenate all the DataFrames in the list
    df = pd.concat(dfs, ignore_index=True)

    return df

def dice_coeff(y_true, y_pred):
    """
    Calcule le coefficient de Dice entre les véritables masques d'image et les masques prédits.

    Paramètres:
    y_true (numpy.ndarray): Un tableau numpy contenant les véritables masques d'image. Chaque masque est un tableau numpy 2D ou 3D.
    y_pred (numpy.ndarray): Un tableau numpy contenant les masques d'image prédits. Chaque masque est un tableau numpy 2D ou 3D.

    Retourne:
    float: Le coefficient de Dice entre les véritables masques d'image et les masques prédits. Le coefficient de Dice est une mesure de la similarité entre deux ensembles, et varie de 0 (pas de recouvrement) à 1 (recouvrement parfait).

    Note:
    La fonction aplatit les masques d'image avant de calculer le coefficient de Dice pour éviter les problèmes de forme. Un terme de lissage est ajouté au numérateur et au dénominateur du coefficient de Dice pour éviter la division par zéro.
    """
    smooth = 1.0
    y_true_f = K.cast(K.flatten(y_true), K.floatx())
    y_pred_f = K.cast(K.flatten(y_pred), K.floatx())
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    """
    Calcule la perte de Dice entre les véritables masques d'image et les masques prédits.

    Paramètres:
    y_true (numpy.ndarray): Un tableau numpy contenant les véritables masques d'image. Chaque masque est un tableau numpy 2D ou 3D.
    y_pred (numpy.ndarray): Un tableau numpy contenant les masques d'image prédits. Chaque masque est un tableau numpy 2D ou 3D.

    Retourne:
    float: La perte de Dice entre les véritables masques d'image et les masques prédits. La perte de Dice est 1 moins le coefficient de Dice, donc elle varie de 0 (recouvrement parfait) à 1 (pas de recouvrement).

    Note:
    La fonction utilise la fonction dice_coeff pour calculer le coefficient de Dice.
    """
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def total_loss(y_true, y_pred):
    """
    Calcule la perte totale entre les véritables masques d'image et les masques prédits.

    Paramètres:
    y_true (numpy.ndarray): Un tableau numpy contenant les véritables masques d'image. Chaque masque est un tableau numpy 2D ou 3D.
    y_pred (numpy.ndarray): Un tableau numpy contenant les masques d'image prédits. Chaque masque est un tableau numpy 2D ou 3D.

    Retourne:
    float: La perte totale entre les véritables masques d'image et les masques prédits. La perte totale est la somme de l'entropie croisée catégorielle et de trois fois la perte de Dice.

    Note:
    La fonction utilise la fonction dice_loss pour calculer la perte de Dice et la fonction categorical_crossentropy de Keras pour calculer l'entropie croisée catégorielle.
    """
    loss = categorical_crossentropy(y_true, y_pred) + (3 * dice_loss(y_true, y_pred))
    return loss


def train_model(
    model,
    model_name,
    data_train,
    data_train_augmented,
    data_val_augmented,
    data_val,
    X_test,
    loss_function,
    augmented=False,
):
    """
    Calcule la perte totale entre les véritables masques d'image et les masques prédits.

    Paramètres:
    y_true (numpy.ndarray): Un tableau numpy contenant les véritables masques d'image. Chaque masque est un tableau numpy 2D ou 3D.
    y_pred (numpy.ndarray): Un tableau numpy contenant les masques d'image prédits. Chaque masque est un tableau numpy 2D ou 3D.

    Retourne:
    float: La perte totale entre les véritables masques d'image et les masques prédits. La perte totale est la somme de l'entropie croisée catégorielle et de trois fois la perte de Dice.

    Note:
    La fonction utilise la fonction dice_loss pour calculer la perte de Dice et la fonction categorical_crossentropy de Keras pour calculer l'entropie croisée catégorielle.
    """
    train_times = {}
    filepath_check = CHECKPOINT_PATH + model_name + "/"
    model_path = CHECKPOINT_PATH + model_name
    history_path_file = HISTORY_PATH + model_name + "_history.pkl"

    if os.path.exists(model_path):
        if loss_function == "categorical_crossentropy":
            model = load_model(model_path)
        else:
            model = load_model(
                model_path,
                custom_objects={"dice_loss": dice_loss, "total_loss": total_loss},
                compile=True,
            )
        print(f"Récupération du modèle {model.name}")

        training_time = train_times[model_name]

        with open(history_path_file, "rb") as f:
            history = pickle.load(f)
        hist_df = pd.DataFrame.from_dict(history)

    else:
        start_train = time.time()

        # compile
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss_function,
            metrics=[OneHotMeanIoU(num_classes=NB_CLASSE, name="mean_iou")],
        )

        mc = ModelCheckpoint(
            mode="max",
            filepath=filepath_check,
            monitor=monitor_val,
            save_best_only="True",
            save_weights_only="True",
        )
        es = EarlyStopping(
            mode="max", monitor=monitor_val, patience=patience, verbose=1
        )

        callbacks = [mc, es]

        # fit
        if augmented:
            history = model.fit(
                data_train_augmented,
                epochs=nbr_epochs,
                validation_data=data_val_augmented,
                use_multiprocessing=False,
                workers=2,
                callbacks=callbacks,
            )
        else:
            history = model.fit(
                data_train,
                epochs=nbr_epochs,
                validation_data=data_val,
                use_multiprocessing=False,
                workers=2,
                callbacks=callbacks,
            )

        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)

        # save history
        with open(history_path_file, mode="wb") as f:
            pickle.dump(hist_df, f)

        # Save the model
        model.load_weights(filepath_check)
        model.save(model_path)

        # Convert the model to TFLite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the TFLite model to a file
        tflite_model_path = model_path + ".tflite"
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)

        end_train = time.time()
        training_time = end_train - start_train
        train_times[model_name] = training_time

        with open(model_path + "train_times.pkl", "wb") as f:
            pickle.dump(train_times, f)

    return model, hist_df, training_time


def predict_model(
    model, model_name, X_test, data_val, loss_type, training_time, show_pred=False
):
    """
    Effectue des prédictions avec un modèle de segmentation d'image, évalue le modèle sur un ensemble de validation, et affiche les prédictions si demandé.

    Paramètres:
    model (tf.keras.Model): Le modèle à utiliser pour les prédictions.
    model_name (str): Le nom du modèle.
    X_test (numpy.ndarray): Les données de test.
    data_val (numpy.ndarray): Les données de validation.
    loss_type (str): Le type de fonction de perte utilisée pour l'entraînement du modèle.
    training_time (float): Le temps d'entraînement du modèle.
    show_pred (bool): Si True, affiche les prédictions pour les 10 premières images de test. Par défaut, False.

    Retourne:
    tuple: Un tuple contenant les prédictions sous forme d'un tableau numpy, et une liste de dictionnaires contenant les résultats de l'évaluation du modèle et les temps d'entraînement et de prédiction.

    Note:
    La fonction évalue d'abord le modèle sur l'ensemble de validation, puis effectue des prédictions sur l'ensemble de test. Les prédictions sont retournées sous forme d'un tableau numpy où chaque élément est l'indice de la classe prédite pour chaque pixel. Si show_pred est True, la fonction affiche également les prédictions pour les 10 premières images de test.
    """
    loss_score, mean_iou_score = model.evaluate(data_val)
    print("Pour le meilleur modèle on obtient :")
    print("mean_iou :", mean_iou_score)
    print("loss :", loss_score)

    start_pred = time.time()

    y_pred = model.predict(X_test)
    y_pred_argmax = np.argmax(y_pred, axis=3)

    end_pred = time.time()
    predict_time = end_pred - start_pred

    if show_pred:
        for index in range(10):
            print(index)
            plt.imshow(X_test[index])
            plt.imshow(y_pred_argmax[index], alpha=0.4)
            plt.show()

    results = []
    results.append(
        {
            "model_name": model_name,
            "loss_type": loss_type,
            "loss_score": loss_score,
            "mean_iou_score": mean_iou_score,
            "training_time": training_time,
            "predict_time": predict_time,
        }
    )

    return y_pred_argmax, results
