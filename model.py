from keras.applications import vgg19
from keras import backend as K
from utils import *

'''
FUNCTION OF MATRIX CALCULATION AND LOSSES (CONTENT, STYLE)
'''
# Gram Matrix for style loss function
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def content_loss(content, synthesized):
    return K.sum(K.square(synthesized - content))

def style_loss(style, synthesized, size):
    style = gram_matrix(style)
    synthesized = gram_matrix(synthesized)
    channels = 3
    return K.sum(K.square(style - synthesized) / (4. * (channels ** 2) * (size ** 2)))

# Load model (VGG19 with imagenet)
def load_model(content, style, size):
    target_image = K.constant(preprocessing_image(content))
    style_image = K.constant(preprocessing_image(style))
    synthesized_image = K.placeholder((1, 400, size, 3))

    input_tensor = K.concatenate([target_image,
                                  style_image,
                                  synthesized_image], axis=0)

    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet',
                        include_top=False)
    print('Model loaded')
    return model, synthesized_image


