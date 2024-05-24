from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPool2D,
    Conv2DTranspose,
    Concatenate,
    Input,
    MaxPooling2D,
    Dropout,
    Reshape,
    Lambda,
    UpSampling2D,
    AveragePooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, ResNet50
import tensorflow as tf


# Convolutional block
def conv_block(input, num_filters):
    """
    Applique une couche de convolution à l'entrée donnée.

    Args:
        input (tensor): Le tensor d'entrée.
        num_filters (int): Le nombre de filtres à utiliser dans la couche de convolution.

    Returns:
        tensor: Le tensor de sortie après l'application de la couche de convolution, de la normalisation par lots et de l'activation ReLU.
    """
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# Encoder block
def encoder_block(input, num_filters):
    """
    Cette fonction définit un bloc décodeur dans l'architecture U-Net.

    Args:
        input (tf.Tensor): Le tensor d'entrée pour le bloc décodeur.
        skip_features (tf.Tensor): Le tensor de connexion latérale provenant du bloc encodeur correspondant.
        num_filters (int): Le nombre de filtres pour les couches convolutionnelles.

    Returns:
        x (tf.Tensor): Le tensor de sortie après l'application d'un bloc convolutionnel.
    """
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


# Decoder block
def decoder_block(input, skip_features, num_filters):
    """
    Cette fonction définit un bloc décodeur dans l'architecture U-Net.

    Args:
        input (tf.Tensor): Le tensor d'entrée pour le bloc décodeur.
        skip_features (tf.Tensor): Le tensor de connexion latérale provenant du bloc encodeur correspondant.
        num_filters (int): Le nombre de filtres pour les couches convolutionnelles.

    Returns:
        x (tf.Tensor): Le tensor de sortie après l'application d'un bloc convolutionnel.
    """
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


# Build Mini U-Net
def build_mini_unet(input_shape, n_classes):
    """
    Cette fonction construit un modèle Mini U-Net pour la segmentation d'images.

    Args:
        input_shape (tuple): La forme de l'entrée sous la forme (hauteur, largeur, canaux).
        n_classes (int): Le nombre de classes pour la classification.

    Returns:
        model (tf.keras.Model): Le modèle Mini U-Net.
    """
    inputs = Input(input_shape)

    # Encoder blocks
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)

    # Bridge
    b1 = conv_block(p2, 128)

    # Decoder blocks
    d1 = decoder_block(b1, s2, 64)
    d2 = decoder_block(d1, s1, 32)

    # Output layer
    if n_classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d2)

    model = Model(inputs, outputs, name="Mini_U-Net")
    return model

from tensorflow.keras.layers import AveragePooling2D, Conv2D, UpSampling2D, Concatenate

def psp_block(input_tensor, n_filters):
    """
    This function builds a Pyramid Scene Parsing (PSP) block.

    Args:
        input_tensor (tf.Tensor): The input tensor.
        n_filters (int): The number of filters for the convolutional layers.

    Returns:
        output_tensor (tf.Tensor): The output tensor.
    """
    # Pyramid pooling with different window sizes
    pool_sizes = [1, 2, 4, 8]
    pooled_tensors = []
    for pool_size in pool_sizes:
        pooled = AveragePooling2D(pool_size=(pool_size, pool_size))(input_tensor)
        conv = Conv2D(n_filters, 1, padding="same", activation="relu")(pooled)
        upsampled = UpSampling2D(size=(pool_size, pool_size))(conv)
        pooled_tensors.append(upsampled)

    # Concatenate the original tensor with the pooled tensors
    output_tensor = Concatenate(axis=-1)([input_tensor] + pooled_tensors)
    return output_tensor

# Modify the U-Net architecture to incorporate a PSP block
def build_psp_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    # Encoder blocks
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)

    # PSP block
    psp = psp_block(p2, 128)

    # Decoder blocks
    d1 = decoder_block(psp, s2, 64)
    d2 = decoder_block(d1, s1, 32)

    # Output layer
    if n_classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"
    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d2)

    model = Model(inputs, outputs, name="PSP_U-Net")
    return model


def build_vgg16_unet(input_shape, nb_classes):
    """
    Cette fonction construit un modèle U-Net basé sur VGG16 pour la segmentation d'images.

    Args:
        input_shape (tuple): La forme de l'entrée sous la forme (hauteur, largeur, canaux).
        nb_classes (int): Le nombre de classes pour la classification.

    Returns:
        model (tf.keras.Model): Le modèle U-Net basé sur VGG16.
    """

    """Input"""
    inputs = Input(input_shape)

    """ Pre-trained VGG16 Model """
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output  ## (512 x 512)
    s2 = vgg16.get_layer("block2_conv2").output  ## (256 x 256)
    s3 = vgg16.get_layer("block3_conv3").output  ## (128 x 128)
    s4 = vgg16.get_layer("block4_conv3").output  ## (64 x 64)

    """ Bridge """
    b1 = vgg16.get_layer("block5_conv3").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)  ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)  ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)  ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)  ## (512 x 512)

    if nb_classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"

    """ Output """
    outputs = Conv2D(nb_classes, 1, padding="same", activation=activation)(d4)

    model = Model(inputs, outputs, name="VGG16_U-Net")
    return model


def build_resnet50(input_shape, nb_classes):
    """
    Cette fonction construit un modèle U-Net basé sur ResNet50 pour la segmentation d'images.

    Args:
        input_shape (tuple): La forme de l'entrée sous la forme (hauteur, largeur, canaux).
        nb_classes (int): Le nombre de classes pour la classification.

    Returns:
        model (tf.keras.Model): Le modèle U-Net basé sur ResNet50.
    """

    """Input"""
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = resnet50.layers[0].output  ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output  ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)  ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)  ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)  ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)  ## (512 x 512)

    if nb_classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"

    """ Output """
    outputs = Conv2D(nb_classes, 1, padding="same", activation=activation)(d4)

    model = Model(inputs, outputs, name="ResNet50_U-Net")
    return model

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model

def build_vgg19_unet(input_shape, nb_classes):
    """
    This function builds a U-Net model based on VGG19 for image segmentation.

    Args:
        input_shape (tuple): The shape of the input in the form (height, width, channels).
        nb_classes (int): The number of classes for classification.

    Returns:
        model (tf.keras.Model): The U-Net model based on VGG19.
    """

    """Input"""
    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg19.get_layer("block1_conv2").output  ## (512 x 512)
    s2 = vgg19.get_layer("block2_conv2").output  ## (256 x 256)
    s3 = vgg19.get_layer("block3_conv4").output  ## (128 x 128)
    s4 = vgg19.get_layer("block4_conv4").output  ## (64 x 64)

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)  ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)  ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)  ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)  ## (512 x 512)

    if nb_classes == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"

    """ Output """
    outputs = Conv2D(nb_classes, 1, padding="same", activation=activation)(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model