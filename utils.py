from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
import numpy as np
from keras import backend as K

'''
FUNCTION OF IMAGE LOADING AND PROCESSING
'''
# Load Images (content, style reference)
def load_images(content_image_path, style_image_path, img_height=400):
    width, height = load_img(content_image_path).size()
    img_width = int(width * img_height / height)
    content_image = load_img(content_image_path, target_size=(img_height, img_width))
    style_image = load_img(style_image_path, target_size=(img_height, img_width))
    size = img_height * img_width
    return content_image, style_image, size


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