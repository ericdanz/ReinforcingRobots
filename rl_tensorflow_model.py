import tensorflow as tf
import numpy

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class ActionLearner(object):
    def __init__(self, n_filters, n_hidden, n_out):

        self.x = tf.placeholder("float", shape=[None, 64,64,3])
        self.y = tf.placeholder("float", shape=[None, n_out])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        W_conv1 = weight_variable([5, 5, 3, n_filters])
        b_conv1 = bias_variable([n_filters])
        h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_fc1 = weight_variable([32*32*n_filters, n_hidden])
        b_fc1 = bias_variable([n_hidden])
        h_pool1_flat = tf.reshape(h_pool1, [-1, 32*32*n_filters])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_keep_prob)

        W_fc2 = weight_variable([n_hidden, n_out])
        b_fc2 = bias_variable([n_out])
        self.output=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.predictions = tf.argmax(self.output, 1)
        # idx = numpy.nonzero(self.y)
        # old_y = tf.identity(self.y[idx])
        # self.y = tf.identity(self.output)
        # self.y[idx] = old_y
        self.single_action_cost = tf.reduce_mean(tf.pow((self.output - self.y),2))
        correct_prediction = tf.equal(tf.argmax(self.output,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        self.number_of_actions = n_out

    def return_action(self,screen):
        return 0
