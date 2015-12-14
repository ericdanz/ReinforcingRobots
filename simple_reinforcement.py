from rl_tensorflow_model import ActionLearner
import numpy
from simulation_simulator import Simulator
from make_states import make_states,make_one_set
import random
import tensorflow as tf
import datetime
import cv2
import getopt,sys,argparse
import time

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--image_size",help="size of simulation screen",default=64,type=int)
    parser.add_argument("-b","--batch_size",help="batch size for training",default=64,type=int)
    parser.add_argument("-g","--gpu",help="which gpu to use (-1 for cpu)",default=-1,type=int)
    parser.add_argument("--number_of_games",help="how many games to simulate per training run",default=100,type=int)

    args = parser.parse_args()

    image_size = args.image_size
    number_of_games = args.number_of_games
    batch_size = args.batch_size
    gpu_flag = args.gpu
    print(image_size,batch_size)

    learning_rate = 1e-3
    #epsilon is the decision parameter - do you use the actor's actions or do them randomly?
    epsilon = 1
    epsilon_decay = 0.04
    display_steps = 200
    sim = Simulator(image_size,2)
    if gpu_flag > -1:
        device_string = '/gpu:{}'.format(gpu_flag)
    else:
        device_string = "/cpu:0"
    with tf.Graph().as_default(), tf.device(device_string):
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))
        with sess.as_default():
            learner = ActionLearner(
                image_size=sim.image_size,
                n_filters=64,
                n_hidden=1024,
                n_out=4
                )
            learner.set_sess(sess)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(learner.single_action_cost)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            loss_summary = tf.scalar_summary("cost", learner.single_action_cost)
            #visualize those first level filters
            filter_summary = tf.image_summary("filters",learner.first_level_filters,max_images=6)

            saver = tf.train.Saver(tf.all_variables())
            sess.run(tf.initialize_all_variables())
            summary_writer = tf.train.SummaryWriter('/tmp/logs', sess.graph_def)

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  learner.x: x_batch,
                  learner.y: y_batch,
                  learner.dropout_keep_prob: 0.5
                }
                _, step,  loss, test_diff,loss_summ,filter_summ = sess.run(
                    [train_op, global_step,  learner.single_action_cost, learner.test_diff,loss_summary,filter_summary],
                    feed_dict)
                summary_writer.add_summary(loss_summ, step)
                if step % 50 == 0:
                    summary_writer.add_summary(filter_summ, step)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {}".format(time_str, step, loss))
                # print("{}".format(test_diff))
            for i in range(1000):
                current_epsilon = epsilon - epsilon_decay*i

                #create a batch of states
                state_list,avg_game_lengths = make_states(sim,learner,current_epsilon,number_of_steps=10,number_of_games=number_of_games)

                #create a random selection of this state list for training
                screens = numpy.zeros((batch_size,sim.image_size,sim.image_size,3))
                actions = numpy.zeros((batch_size,4),dtype=numpy.float32)
                for j in range(10):
                    #grab random batches from the training images
                    random_states = random.sample(state_list,batch_size)
                    index = 0
                    for state in random_states:
                        screens[index,:,:,:] = state[0][0]
                        actions[index,state[0][2]] = float(state[0][1])
                        index += 1
                    # screens = screens - numpy.mean(numpy.mean(screens,axis=0),axis=0)
                    train_step(screens,actions)

                current_step = tf.train.global_step(sess, global_step)
                game_length_summary = tf.scalar_summary("game_length",avg_game_lengths)
                game_length_summ = sess.run(game_length_summary)
                summary_writer.add_summary(game_length_summ, current_step)

                if current_step % display_steps == 0:

                    #do a test run
                    sim.reset(sim.image_size,10)
                    #get an average game length, as proxy for learnin'
                    game_lengths = []
                    for j in range(20):
                        display_state_list = make_one_set(sim,learner,0,number_of_steps=10)
                        game_lengths.append(len(display_state_list))
                    print("The average game length (lower is better, and 10 is the max): {}".format(numpy.mean(game_lengths)))

                    #show the last one
                    for state in display_state_list:
                        cv2.imshow('sim',cv2.resize(state[0][0],(0,0),fx=2,fy=2))
                        cv2.waitKey(1000)
                    cv2.destroyAllWindows()
