"""
This module contains functions to run the neural network developed here:
https://github.com/iro-cp/FCRN-DepthPrediction/
It works with the numpy model provided by them.
The network can be started with the start_network function
The network can then supply generated depth images using the predict_from_image function.
NOTE: Still uses outdated tensorflow functions and compatibility settings.
"""


import argparse
import collections
import numpy as np
from PIL import Image
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf
import models


tf.compat.v1.disable_eager_execution()

def start_network(model_data_path):
    """
    Isolated function to start the network and return it as a stucture.
    This is to make it re-runnable on multiple images using predict_from_image.
    """
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Create a placeholder for the input image
    input_node = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False, trainable=False)

    sess = tf.compat.v1.Session()

    # Load the converted parameters
    print('Loading the model')

    net.load(model_data_path, sess)

    #using a namedtuple to return the network state.
    Network = collections.namedtuple('Network', ['sess', 'net', 'input_node'])
    network = Network(sess, net, input_node)
    print('returning the model')

    return network

def predict_from_image(network, image):
    """
    Simplified version of the predict function.
    This has the running of the network separated from starting the network.
    This is to process multiple images without reloading 4GBs of 'network'
    """

    # Default input size
    height = 228
    width = 304

    # read cv image
    img = Image.fromarray(image)
    img = img.resize((width, height), Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis=0)

    # Evalute the network for the given image
    pred = network.sess.run(network.net.get_output(), feed_dict={network.input_node: img[:, :, :]})

    # Convert 1 or 2d sequence of scalars to RGBA array.
    # https://matplotlib.org/3.1.1/api/cm_api.html#matplotlib.cm.ScalarMappable
     # bytes = true makes uint8
    out_img = plt.cm.ScalarMappable().to_rgba(
        pred[0, :, :, 0], alpha=None, bytes=True, norm=True)

    return out_img

def predict(model_data_path, image_path):
    """
    The main predict function present in the original source file.
    It either opens a file or interprets a given image for its input.
    Then it sets up the neural network, runs a prediction, and plots its output.
    """

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Read image
    img = Image.open(image_path)
    img = img.resize([width, height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis=0)

    # read cv image
    # img = Image.fromarray(image_path)
    # img = img.resize([width,height], Image.ANTIALIAS)
    # img = np.array(img).astype('float32')
    # img = np.expand_dims(np.asarray(img), axis=0)

    # Create a placeholder for the input image
    input_node = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    sess = tf.compat.v1.Session()

    # Load the converted parameters
    print('Loading the model')

    # Use to load from ckpt file
    # saver = tf.compat.v1.train.Saver()
    # saver.restore(sess, model_data_path)
    # Use to load from npy file
    net.load(model_data_path, sess)

    # Evalute the network for the given image
    pred = sess.run(net.get_output(), feed_dict={input_node: img[:, :, :]})

    # Plot result
    fig = plt.figure()
    plt_im = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
    fig.colorbar(plt_im)
    plt.show()

    return pred

def prep_image(image):
    """
    Sets image to the correct size for comparison, e.g. 228 x 304, the output of the NN.
    then, converts it to grayscale to make it one channel that needs comparison.
    """
    image = cv.resize(src=image, dsize=(228, 304), dst=image, interpolation=cv.INTER_CUBIC)

    print('resized dimensions = ')
    print(image.shape)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    print('gray converted = ')
    print(gray.shape)

    return gray

def load_image():
    """ Imreads a prompted image path. """

    print('please provide the path to an image: ')
    image_path = input()
    image = cv.imread(image_path)

    return image


# Mean Squared Error comparison on openCV images.
# From: https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
def mse(image_a, image_b):
    """
    the 'Mean Squared Error' between the two images is the
    sum of the squared difference between the two images;
    In other words, it is absolute difference in pixel value at every pixel,
    devided by the size of the image. (average / mean)
    NOTE: the two images must have the same dimension
    """

    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    return err

def compare_images(image_a, image_b):
    """
    function for comparing two images.
    It calculates the MSE (Mean Squared Error) and SSIM (Structural Simularity)
    which it prints and returns in that order.
    """

    mean_error = mse(image_a, image_b)
    struc_sim = ssim(image_a, image_b)

    print('mse comparison result: ')
    print(mean_error)

    print('ssim (Structural Simularity) comparison result: ')
    print(struc_sim)

    return mean_error, struc_sim

def open_webcam(network):
    """
    This function opens a webcam, provides its images to the network and displays its output.
    """

    vid_capture = cv.VideoCapture(0)
    cv.namedWindow("cam", cv.WINDOW_NORMAL)
    cv.namedWindow("net", cv.WINDOW_NORMAL)

    while True:

        # Capture each frame of webcam video

        if vid_capture.read():

            ret, frame = vid_capture.read()
            cv.imshow("cam", frame)

            frame = predict_from_image(network, frame)
            cv.imshow("net", frame)

            # Close and break the loop after pressing "x" key
            if cv.waitKey(1) &0XFF == ord('x'):
                break


def main():
    """
    main function where the model file is parsed,
    the network is started and the main loop for processing images begins.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    args = parser.parse_args()

    network = start_network(args.model_path)
    open_webcam(network)

#    while True:
#
#        print('RGB image:')
#        rgb_image = load_image()
#
#        print('Depth image:')
#        depth_image = prep_image(load_image())
#
#        pred_image = prep_image(predict_from_image(network, rgb_image))
#
#        compare_images(depth_image, pred_image)
#        while True:
#            cv.imshow("true depth", depth_image)
#            cv.imshow("pred depth", pred_image)
#
#            if cv.waitKey(1) & 0xFF == ord('q'):
#                break

if __name__ == '__main__':
    main()
