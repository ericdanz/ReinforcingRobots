import tensorflow as tf
import numpy

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name='weights')

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name='biases')

def leakyReLU(x):
    return tf.nn.relu(x) - 0.01 * tf.nn.relu(-x)

class ActionLearner(object):
    def __init__(self,image_size, n_filters, n_hidden, n_out):
        self.image_size = image_size
        self.display_output = numpy.zeros(n_out)
        with tf.name_scope('model')  as scope:
            self.x = tf.placeholder("float", shape=[None, self.image_size,self.image_size,3],name='input_x')
            self.y = tf.placeholder("float", shape=[None, n_out],name='input_y')
            self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')


            with tf.name_scope('hidden_conv1') as scope:
                W_conv1 = weight_variable([5, 5, 3, n_filters])
                b_conv1 = bias_variable([n_filters])
                #h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
                h_conv1 = leakyReLU(conv2d(self.x, W_conv1) + b_conv1)
                h_pool1 = max_pool_2x2(h_conv1)

            with tf.name_scope('hidden_conv2') as scope:
                W_conv2 = weight_variable([3, 3, n_filters, n_filters*2])
                b_conv2 = bias_variable([n_filters*2])
                #h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
                h_conv2 = leakyReLU(conv2d(h_pool1, W_conv2) + b_conv2)
                h_pool2 = max_pool_2x2(h_conv2)

            with tf.name_scope('hidden_conv3') as scope:
                W_conv3 = weight_variable([3, 3, n_filters*2, n_filters*2])
                b_conv3 = bias_variable([n_filters*2])
                #h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
                h_conv3 = leakyReLU(conv2d(h_pool2, W_conv3) + b_conv3)
                h_pool3 = max_pool_2x2(h_conv3)

            with tf.name_scope('hidden_fc1') as scope:
                W_fc1 = weight_variable([int(self.image_size/8)*int(self.image_size/8)*n_filters*2, n_hidden])
                b_fc1 = bias_variable([n_hidden])
                h_pool3_flat = tf.reshape(h_pool3, [-1, int(self.image_size/8)*int(self.image_size/8)*n_filters*2])
                #h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
                h_fc1 = leakyReLU(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
                h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_keep_prob)

            with tf.name_scope('hidden_fc2') as scope:
                W_fc2 = weight_variable([n_hidden, n_out])
                b_fc2 = bias_variable([n_out])

            # L2 regularization for all parameters.
            regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
                tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) +
                tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))

            with tf.name_scope('output') as scope:
                self.output=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
                self.predictions = tf.argmax(self.output, 1)

                #attempt to figure out the correct way to do the loss
                #the y matrix is sparse, with only the actions taken having scores
                #only those scores should be compared with the predicted scores,
                #and the difference between the y and output matrix should be 0
                #everywhere else
                y_2 = tf.identity(self.y)  #this copies y
                y_2 = y_2 / (y_2 + 1e-8) #should be only ones
                y_2 =  y_2* self.output #should only be the scores at the y actions
                output_2 = tf.identity(self.output)
                output_2 = output_2 - y_2 #should now have 0s at the nonzero y
                output_2 = output_2 + self.y #should now have y values at the actions taken, output values everywhere else

                self.normal_cost = tf.reduce_mean(tf.pow((self.output - self.y),2)) #this just subtracts the largely 0 y matrix
                self.single_action_cost = tf.reduce_mean(tf.pow((self.output - output_2),2)) #this subtracts a matrix almost identical to self.output
                #add l2 penalty
                self.single_action_cost += regularizers*1e-7
                correct_prediction = tf.equal(tf.argmax(self.output,1), tf.argmax(self.y,1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                self.number_of_actions = n_out
                self.first_level_filters = tf.transpose(tf.identity(W_conv1),perm=[3,0,1,2])

    def set_sess(self,session):
        self.sess = session

    def return_action(self,screen):
        input_screen = screen.view()
        input_screen.shape = (1,self.image_size,self.image_size,3)
        feed_dict = {
          self.x: input_screen,
          self.dropout_keep_prob: 1.0
          }
        action,display_output  = self.sess.run(
            [self.predictions,self.output],
            feed_dict)
        self.display_output = display_output
        return int(action[0])
