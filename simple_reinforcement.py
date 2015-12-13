from rl_tensorflow_model import ActionLearner
import numpy
from simulation_simulator import Simulator
from make_states import make_states,make_one_set
import random
import tensorflow as tf
import datetime
import cv2
import getopt,sys

if __name__=="__main__":
    optlist, args = getopt.getopt(sys.argv[1:],'i')
    image_size = 64
    number_of_games = 100
    batch_size = 50
    gpu_flag = -1
    for n,(o,a) in enumerate(optlist):
        if o in ['-i','--image_size']:
            image_size = int(args[n])
        if o in ['-b','--batch_size']:
            batch_size = int(args[n])
        if o in ['--number_of_games']:
            number_of_games = int(args[n])
        if o in ['-g','--gpu']:
            gpu_flag = int(args[n])

    rng = numpy.random.RandomState(1234)
    learning_rate = 0.01
    L2_reg = 0.0001
    #epsilon is the decision parameter - do you use the actor's actions or do them randomly?
    epsilon = 1
    epsilon_decay = 0.001
    display_steps = 20
    sim = Simulator(image_size,10)
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True))
        with sess.as_default():
            learner = ActionLearner(
                image_size=sim.image_size,
                n_filters=32,
                n_hidden=1024,
                n_out=4
                )
            learner.set_sess(sess)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)
            if gpu_flag > -1:
                device_string = '/gpu:{}'.format(gpu_flag)
            else:
                device_string = "/cpu:0"
            with tf.device(device_string):
                with tf.name_scope('gpu') as scope:
                    grads_and_vars = optimizer.compute_gradients(learner.single_action_cost)
                    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            saver = tf.train.Saver(tf.all_variables())
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  learner.x: x_batch,
                  learner.y: y_batch,
                  learner.dropout_keep_prob: 0.5
                }
                _, step,  loss, accuracy = sess.run(
                    [train_op, global_step,  learner.single_action_cost, learner.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {}".format(time_str, step, loss))

            for i in range(1000):
                current_epsilon = epsilon - epsilon_decay*i
                #create a batch of states
                state_list = make_states(sim,learner,current_epsilon,number_of_steps=10,number_of_games=number_of_games)
                #create a random selection of this state list for training
                screens = numpy.zeros((batch_size,sim.image_size,sim.image_size,3))
                actions = numpy.zeros((batch_size,4),dtype=numpy.float32)
                states = random.sample(state_list,batch_size)
                index = 0
                for state in states:
                    screens[index,:,:,:] = state[0][0]
                    actions[index,state[0][2]] = float(state[0][1])
                    index += 1
                print(screens.shape,actions.shape)

                train_step(screens,actions)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % display_steps == 0:
                    #do a test run
                    sim.reset(sim.image_size,10)
                    display_state_list = make_one_set(sim,learner,0,number_of_steps=10)
                    for state in display_state_list:
                        print(state[0][0].shape)
                        cv2.imshow('sim',state[0][0])
                        cv2.waitKey(500)