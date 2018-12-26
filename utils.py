from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
import numpy as np
from keras import backend as K

'''
FUNCTION OF IMAGE LOADING AND PROCESSING
'''
# Load Images (target, style reference)
def load_images(target_image_path, style_image_path, img_height=400):
    width, height = load_img(target_image_path).size()
    img_width = int(width * img_height / height)
    target_image = load_img(target_image_path, target_size=(img_height, img_width))
    style_image = load_img(style_image_path, target_size=(img_height, img_width))
    size = img_height * img_width
    return target_image, style_image, size


# Preprocessing & Deprocessing image
def preprocessing_image(image):
    img = img_to_array(image)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

'''
FUNCTION OF MATRIX CALCULATION AND LOSSES (CONTENT, STYLE)
'''
# Gram Matrix for style loss function
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def content_loss(target, synthesized):
    return K.sum(K.square(synthesized - target))

def style_loss(style, synthesized, size):
    style = gram_matrix(style)
    synthesized = gram_matrix(synthesized)
    channels = 3
    return K.sum(K.square(style - synthesized) / (4. * (channels ** 2) * (size ** 2)))