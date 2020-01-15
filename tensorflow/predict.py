import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import models
import cv2 as cv
import collections
import scipy.io
from skimage.metrics import structural_similarity as ssim

tf.compat.v1.disable_eager_execution()

def startNetwork(model_data_path):

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Create a placeholder for the input image
    input_node = tf.compat.v1.placeholder(dtype=tf.float32, shape=( 1, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False, trainable=False)

    sess = tf.compat.v1.Session()

    # Load the converted parameters
    print('Loading the model')

    net.load(model_data_path, sess) 

    #using a namedtuple to return the network state.
    Network = collections.namedtuple('Network', ['sess', 'net', 'input_node'] )
    network = Network(sess, net, input_node)
    print('returning the model')
    
    return network

def predictFromImage(network, image):

    # Default input size
    height = 228
    width = 304
   
    # read cv image
    img = Image.fromarray(image)
    img = img.resize( (width,height) , Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
   
    # Evalute the network for the given image
    pred = network.sess.run(network.net.get_output(), feed_dict={ network.input_node: img[:,:,:]})
    
    # Convert 1 or 2d sequence of scalars to RGBA array.
    # https://matplotlib.org/3.1.1/api/cm_api.html#matplotlib.cm.ScalarMappable 
    out_img = plt.cm.ScalarMappable().to_rgba( pred[0,:,:,0], alpha=None, bytes=True, norm=True) # bytes = true makes uint8

    return out_img 

#v1.variable_scope()
def predict(model_data_path, image_path):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = Image.open(image_path)
    img = img.resize([width, height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)

    # read cv image
    # img = Image.fromarray(image_path)
    # img = img.resize([width,height], Image.ANTIALIAS)
    # img = np.array(img).astype('float32')
    # img = np.expand_dims(np.asarray(img), axis = 0)
   
    # Create a placeholder for the input image
    input_node = tf.compat.v1.placeholder(dtype=tf.float32, shape=( 1, height, width, channels))

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
    pred = sess.run(net.get_output(), feed_dict={input_node: img[:,:,:]})
    
    # Plot result
    fig = plt.figure()
    ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
    fig.colorbar(ii)
    plt.show()
    
    return pred


def nop():
    return

# Sets image to the correct size for comparison, e.g. 228 x 304, the output of the NN.
# then, converts it to grayscale to make it one channel that needs comparison.
def prepImage(image):

    image = cv.resize( src=image, dsize=(228, 304), dst=image, interpolation=cv.INTER_CUBIC )

    print('resized dimensions = ') 
    print(image.shape)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    print('gray converted = ') 
    print(gray.shape)

    return gray

def loadImage():
    print('please provide the path to an image: ')
    imagePath = input()
    image = cv.imread(imagePath)
    return image


# Mean Squared Error comparison on openCV images.
# From: https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
def mse(imageA, imageB):

    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB):

    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    print('mse comparison result: ')
    print(m)

    print('ssim (Structural Simularity) comparison result: ')
    print(s)

    return m , s


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    #parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    network = startNetwork(args.model_path)

    while True:

        print('RGB image:')
        RGBImage = loadImage() 
        
        print('Depth image:')
        DepthImage = prepImage( loadImage() )

        PredImage = prepImage( predictFromImage(network, RGBImage) )

        compare_images(DepthImage, PredImage)
        while True:
            cv.imshow("true depth", DepthImage)
            cv.imshow("pred depth", PredImage)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        




    os._exit(0)

if __name__ == '__main__':
    main()