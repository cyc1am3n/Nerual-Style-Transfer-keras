from keras import backend as K
import numpy as np
from utils import *
from model import *

import argparse

class Evaluator(object):
    def __init__(self, fetch_loss_and_grads, width, height=400,):
        self.img_height = height
        self.img_width = width
        self.fetch_loss_and_grads = fetch_loss_and_grads

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, self.img_height, self.img_height, 3))
        outs = self.fetch_loss_and_grads([x])
        loss_value = outs[0]
        grads_value = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grads_value = grads_value
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_value= np.copy(self.grads_value)
        self.loss_value = None
        self.grads_value = None
        return grad_value

def main(args):
    content_image = args.content_img
    style_image = args.style_img
    img_height = args.img_height
    steps = args.steps
    content_weight = args.c_weight
    style_weight = args.s_weight
    content_layer = 'block' + str(args.c_layer) + '_conv2'
    style_layers = []
    for layer in args.s_layers:
        style_layers.append('block' + str(layer) + '_conv1')

    content_image, style_image, size = load_images(content_image, style_image, img_height=img_height)
    model, synthesized_image = load_model(content_image, style_image, size)
    img_width = size / img_height

    outputs_dict = dict([(layer.name, layer.output) for layer in model.model.layers])

    loss = K.variable(0.)
    layer_features = outputs_dict[content_layer]
    content_image_features = layer_features[0, :, :, :]
    synthesized_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(content_image_features, synthesized_features)

    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_features = layer_features[1, :, :, :]
        synthesized_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, synthesized_features)
        loss += (style_weight / len(style_layers)) * sl

    grads = K.gradients(loss, synthesized_image)[0]

    fetch_loss_and_grads = K.function([synthesized_image], [loss, grads])
    evaluator = Evaluator(fetch_loss_and_grads, img_width, img_height=img_height)

    result_prefix = 'style_transfer_result'



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_img', type=str, default='./datasets/content/cat.jpg')
    parser.add_argument('--style_img', type=str, default='./datasets/style_reference/the_scream.jpg')
    parser.add_argument('--img_height', type=int, default=400,
                        help='the height of synthesized image')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--c_weight', type=int, default=1, help='weight of content reconstruction')
    parser.add_argument('--s_weight', type=int, default=100000, help='weight of style reconstruction')
    parser.add_argument('--c_layer', type=int, default=5,
                        help='layer to use for content reconstruction; default is block5_conv2')
    parser.add_argument('--s_layers', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='layers to use for style reconstruction;'
                             'default is block1_conv1, block2_conv1, block3_conv1, block4_conv1, block5_conv1')

    args = parser.parse_args()
    print(args)
    output = main(args)