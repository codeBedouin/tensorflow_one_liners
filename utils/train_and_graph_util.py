import tensorflow
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data

class train_util(object):
    """"
    class to provide training utilites while working on stuff
    """

    def __init__(self):
        pass
    # put a data feeder
    
    # sess = tf.InteractiveSession()

    # with tfname_scope('input'):
        # x = tf.placeholder(tf.float32, [None, 784], name = 'x-input')
        # y_ = tf.placeholder(tf.float32, [None, 20], name = 'y_out')
    # with tf.name_scope('input_reshape'):
        # image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        # tf.summary.image('input', image_shaped_input, 10)
    def weight_variable(self,shape):
        """weight Variable creation
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(self,shape):
        """bias Variable creation
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    def variable_summaries(self,var):
        """create summaries for visualization
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name.scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def create_fc_layer(self,input_tensor, input_dim, output_dim, layer_name,
                       activation_func=tf.nn.relu):

       """Creating a new fully connected layer
       """
       #TODO: Documentation
        with tf.name_scope(layer_name):
           with tf.name_scope('weights'):
               weights = self.weight_variable([input_dim, output_dim])
               variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activation',preactivate)
            activations = activation_func(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations
    def create_feed_dict(self):
        """function to create feed_dict
        """
        pass

