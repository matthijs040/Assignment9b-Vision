import argparse
import os
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from matplotlib import pyplot as plt
from PIL import Image

import models

import collections


import numpy


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
    Netstuff = collections.namedtuple('netstuff', ['sess', 'net', 'input_node'] )
    netstuff = Netstuff(sess, net, input_node)
    print('returning the model')
    
    return netstuff

#v1.variable_scope()
def predict(model_data_path, image_path):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
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

                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    predict(args.model_path, args.image_paths)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



