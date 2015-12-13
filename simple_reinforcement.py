from rl_tensorflow_model import ActionLearner
import numpy
from simulation_simulator import Simulator
from make_states import make_states
import random
import tensorflow as tf
import datetime

if __name__=="__main__":
    rng = numpy.random.RandomState(1234)
    learning_rate = 0.01
    L2_reg = 0.0001
    #epsilon is the decision parameter - do you use the actor's actions or do them randomly?
    epsilon = 1
    epsilon_decay = 0.001
    sim = Simulator(64,10)
    sess = tf.Session()
    with sess.as_default():
        learner = ActionLearner(
            n_filters=32,
            n_hidden=1024,
            n_out=4
            )

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
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
            print("{}: step {}, loss {}, acc {}".format(time_str, step, loss, accuracy))
        def test_step(x_batch, y_batch):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              learner.x: x_batch,
              learner.y: y_batch,
              learner.dropout_keep_prob: 1.0
            }
            step,  loss, accuracy = sess.run(
                [global_step, learner.single_action_cost, learner.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}, acc {}".format(time_str, step, loss, accuracy))


        for i in range(1000):
            current_epsilon =epsilon - epsilon_decay*i
            #create a batch of states
            state_list = make_states(sim,learner,current_epsilon,number_of_steps=10,number_of_games=100)
            #create a random selection of this state list for training
            screens = numpy.zeros((50,sim.image_size,sim.image_size,3))
            actions = numpy.zeros((50,4),dtype=numpy.float32)
            states = random.sample(state_list,50)
            index = 0
            for state in states:
                screens[index,:,:,:] = state[0][0]
                actions[index,state[0][2]] = float(state[0][1])
                index += 1
            print(screens.shape,actions.shape)

            train_step(screens,actions)
