import tensorflow as tf
import numpy
sess = tf.InteractiveSession()


x = tf.placeholder("float", shape=[None, 64,64,3])
y_ = tf.placeholder("float", shape=[None, 4])


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


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_fc1 = weight_variable([30*30*32, 1024])
b_fc1 = bias_variable([1024])
h_pool1_flat = tf.reshape(h_pool2, [-1, 30*30*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)


W_fc2 = weight_variable([1024, 4])
b_fc2 = bias_variable([4])
y_conv=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
idx = numpy.nonzero(y_)
single_action_cost = (y_conv[idx] - y_[idx])**2
train_step = tf.train.AdamOptimizer(1e-4).minimize(single_action_cost)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
