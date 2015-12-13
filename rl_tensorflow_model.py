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
    def __init__(self,image_size, n_filters, n_hidden, n_out):
        self.image_size = image_size
        self.x = tf.placeholder("float", shape=[None, self.image_size,self.image_size,3],name='input_x')
        self.y = tf.placeholder("float", shape=[None, n_out],name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

        W_conv1 = weight_variable([5, 5, 3, n_filters])
        b_conv1 = bias_variable([n_filters])
        h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)


        W_fc1 = weight_variable([int(self.image_size/2)*int(self.image_size/2)*n_filters, n_hidden])
        b_fc1 = bias_variable([n_hidden])
        h_pool1_flat = tf.reshape(h_pool1, [-1, int(self.image_size/2)*int(self.image_size/2)*n_filters])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_keep_prob)

        W_fc2 = weight_variable([n_hidden, n_out])
        b_fc2 = bias_variable([n_out])

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
            tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))

        self.output=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        self.predictions = tf.argmax(self.output, 1)

        #attempt to figure out the correct way to do the loss
        #the y matrix is sparse, with only the actions taken having scores
        #only those scores should be compared with the predicted scores,
        #and the difference between the y and output matrix should be 0
        #everywhere else
        y_2 = tf.identity(self.y)  #this copies y
        y_2 = y_2 / (y_2 + 1e-8) #should be only ones
        # y_2_mask = tf.to_int64(tf.identity(y_2))
        self.y_2 =  y_2* self.output #should only be the scores at the y actions
        output_2 = tf.identity(self.output)
        output_2 = output_2 - y_2 #should now have 0s at the nonzero y
        output_2 = output_2 + self.y #should now have y values at the actions taken, output values everywhere else
        self.test_diff = self.output-output_2


        self.single_action_cost = tf.reduce_mean(tf.pow((self.output - output_2),2))
        #add l2 penalty
        self.single_action_cost += regularizers*5e-4
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
        action  = self.sess.run(
            [self.predictions],
            feed_dict)

        return int(action[0])
