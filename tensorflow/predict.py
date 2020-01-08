import argparse
import os
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from matplotlib import pyplot as plt
from PIL import Image

import models

import cv2 as cv

import collections



def startNetwork(model_data_path):

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Create a placeholder for the input image
    input_node = tf.compat.v1.placeholder(dtype=tf.float32, shape=( 1, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

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
    img = img.resize( [width,height] , Image.ANTIALIAS)
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


 
def openWebcam( network ):

    vid_capture = cv.VideoCapture(0)
    cv.namedWindow("cam", cv.WINDOW_NORMAL)
    cv.namedWindow("net", cv.WINDOW_NORMAL)

    while(True):

        # Capture each frame of webcam video

        if( vid_capture.read() ):

            ret, frame = vid_capture.read()
            cv.imshow("cam", frame)

            frame = predictFromImage(network, frame)
            cv.imshow("net", frame )

            # Close and break the loop after pressing "x" key
            if cv.waitKey(1) &0XFF == ord('x'):
                break
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    #parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    # predict(args.model_path, args.image_paths)
        
    openWebcam( startNetwork(args.model_path) )
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



