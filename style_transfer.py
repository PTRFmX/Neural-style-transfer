import os
import sys
import scipy.io
import scipy.misc
from PIL import Image
from utils import *
from resizeimage import resizeimage

import argparse
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Default values
CONTENT_WEIGHT = 5
STYLE_WEIGHT = 200
LEARNING_RATE = 2.0
ITERATIONS = 500
PRETRAINED_MODEL_PATH = "pretrained-model/imagenet-vgg-verydeep-19.mat"
RESIZE_IMG_PATH = 'images/resized-style.jpg'

def build_parser():
    """
    Use parser to get user argument.
    """

    parser = argparse.ArgumentParser(description = "Processing user arguments")
    
    parser.add_argument('--content', dest='content', help='content image', metavar='CONTENT', required=True)
    parser.add_argument('--style', dest='style', help='style image', metavar='STYLE', required=True)
    parser.add_argument('--output', dest='output', help='output path', metavar='OUTPUT', required=True)
    parser.add_argument('--model', dest='model', help='model path', metavar='MODEL', default=PRETRAINED_MODEL_PATH)
    parser.add_argument('--iterations', type=int, dest='iterations', help='iterations (default %(default)s)', metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--content-weight', type=float, dest='content_weight', help='content weight (default %(default)s)', metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float, dest='style_weight', help='style weight (default %(default)s)', metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', help='learning rate (default %(default)s)', metavar='LEARNING_RATE', default=LEARNING_RATE)

    return parser

def content_loss(sess, model):
    """
    Compute the content loss.
    """
    def _content_loss(p, x):
        N = p.shape[3]
        M = p.shape[1] * p.shape[2]        
        return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))

    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])

def total_cost(J_content, J_style, alpha, beta):
    """
    Computes the total cost function.
    """
    J = alpha * J_content + beta * J_style
    return J

STYLE_LAYERS = [
    ('conv1_1', 0.5),
    ('conv2_1', 1.0),
    ('conv3_1', 1.5),
    ('conv4_1', 3.0),
    ('conv5_1', 4.0),
]

def style_loss(sess, model):
    """
    Compute the style loss
    """
    def _gram_matrix(F, N, M):
        """
        The gram matrix G.
        """
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x):
        """
        The style loss calculation.
        """
        N = a.shape[3]
        M = a.shape[1] * a.shape[2]
        A = _gram_matrix(a, N, M)
        G = _gram_matrix(x, N, M)
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in STYLE_LAYERS]
    W = [w for _, w in STYLE_LAYERS]
    loss = sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))])
    return loss

def main():

    # Get parser
    parser = build_parser()
    arguments = parser.parse_args()

    # Get all user arguments
    content_img_path = arguments.content
    style_img_path = arguments.style
    iterations = arguments.iterations
    content_weight = arguments.content_weight
    style_weight = arguments.style_weight
    learning_rate = arguments.learning_rate
    model_path = arguments.model
    output_path = arguments.output

    # Start interactive session
    sess = tf.InteractiveSession()

    # Get sizes of content image
    content = Image.open(content_img_path)
    c_width, c_height = content.size

    style = Image.open(style_img_path)

    # Maybe resize style image
    style = style.resize((c_width, c_height))
    style_img_path = RESIZE_IMG_PATH
    style.save(style_img_path)

    # Load content image and style image
    content_image = load_image(content_img_path)
    style_image = load_image(style_img_path)

    # Reset default CONFIG values to actual values
    CONFIG.STYLE_IMAGE = style_image
    CONFIG.CONTENT_IMAGE = content_image

    CONFIG.IMAGE_HEIGHT = content_image.shape[1]
    CONFIG.IMAGE_WIDTH = content_image.shape[2]
    CONFIG.NOISE_RATIO = CONFIG.IMAGE_HEIGHT / CONFIG.IMAGE_WIDTH

    # Generate noise image first
    generated_image = generate_noise_image(content_image)

    model = load_vgg_model(model_path)

    sess.run(tf.global_variables_initializer())

    # Assign the content image to be the input of the VGG model.  
    sess.run(model['input'].assign(content_image))
    J_C = content_loss(sess, model)

    # Assign the style image to be the input of the VGG model.  
    sess.run(model['input'].assign(style_image))
    J_S = style_loss(sess, model)

    # Compute total cost
    J = total_cost(J_C, J_S, content_weight, style_weight)

    # Define optimizer and train_step
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(J)

    def _model_nn(sess, input_image, num_iterations):
        
        # Initialize global variables
        sess.run(tf.global_variables_initializer())
        
        # Run the noisy input image
        sess.run(model['input'].assign(input_image))
        
        for i in range(num_iterations):
        
            # Run the session on the train_step to minimize the total cost
            sess.run(train_step)

            # Compute the generated image
            generated_image = sess.run(model['input'])

            # Print every 20 iteration.
            if i % 20 == 0:

                mixed_image = sess.run(model['input'])
                print("Iteration " + str(i) + " :")
                print('sum : ', sess.run(tf.reduce_sum(mixed_image)))
                print('cost: ', sess.run(J))
                
                # save current generated image
                save_image("output/" + str(i) + ".png", generated_image)
        
        # save last generated image
        save_image(output_path, generated_image)
        
        return generated_image

    _model_nn(sess, generated_image, iterations)

if __name__ == "__main__":
    main()