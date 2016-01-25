from rl_tensorflow_model import ActionLearner
import numpy
from phong_simulator import Simulator
from make_phong_states import make_states,make_one_set
import random
import tensorflow as tf
import datetime
import cv2
import getopt,sys,argparse
import time

#UP_KEY = 63232
#DOWN_KEY = 63233
UP_KEY = 1113938
DOWN_KEY = 1113940
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--image_size",help="size of simulation screen",default=64,type=int)
    parser.add_argument("-b","--batch_size",help="batch size for training",default=128,type=int)
    parser.add_argument("-g","--gpu",help="which gpu to use (-1 for cpu)",default=-1,type=int)
    parser.add_argument("--number_of_games",help="how many games to simulate per training run",default=200,type=int)
    parser.add_argument("--restore",help="to restore from the saved folder",default="No",type=str)
    parser.add_argument("--save_folder",help="where to save the session variables",default="/tmp/logs/",type=str)
    parser.add_argument("--learning_rate",help="how fast to learn",default=1e-5,type=int)
    parser.add_argument("--epsilon_decay",help="how quickly to use the actor in simulations (vs random actions)",default=0.01,type=float)
    parser.add_argument("--display_iterations",help="how often to display a test game",default=100,type=int)
    parser.add_argument("--number_of_filters",help="how many filters the convolutional layer should have",default=32,type=int)
    parser.add_argument("--number_of_hidden",help="how many hidden units to have",default=512,type=int)

    args = parser.parse_args()

    image_size = args.image_size
    number_of_games = args.number_of_games
    batch_size = args.batch_size
    gpu_flag = args.gpu

    learning_rate = args.learning_rate
    #epsilon is the decision parameter - do you use the actor's actions or do them randomly?
    #initially, you want to use random actions - but over time as the actor learns,
    #the actor's actions will be better
    epsilon = 1
    epsilon_decay = args.epsilon_decay
    display_steps = args.display_iterations
    sim = Simulator(1)
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
                n_filters=args.number_of_filters,
                n_hidden=args.number_of_hidden,
                n_out=sim.number_of_actions
                )
            learner.set_sess(sess)

            global_step = tf.Variable(0, name="global_step", trainable=False)

            saver = tf.train.Saver(tf.all_variables())
            sess.run(tf.initialize_all_variables())
            summary_writer = tf.train.SummaryWriter(args.save_folder, sess.graph_def)
            if args.restore != "No":
                saver.restore(sess, args.save_folder+args.restore)


            #just display games
            sim.reset()
            previous_state = numpy.zeros((sim.image_size,sim.image_size,3))
            previous_state[:,:,0] = numpy.reshape(sim.screen,(sim.image_size,sim.image_size))
            screen = sim.screen
            for i in range(1000):
                cv2.imshow('Phong!',cv2.resize(screen,(0,0),fx=2,fy=2))
                key = cv2.waitKey(8)
                if key == UP_KEY:
                    screen,score,points_made,end = sim.do_action(1,side="left")
                elif key == DOWN_KEY:
                    screen,score,points_made,end = sim.do_action(2,side="left")
                else:
                    screen,score,points_made,end = sim.do_action(0,side="left")
                action = learner.return_action(previous_state)
                screen,score,points_made,end = sim.do_action(action)
                previous_state[:,:,1:] = numpy.copy(previous_state[:,:,:2])
                previous_state[:,:,0] = numpy.reshape(screen,(sim.image_size,sim.image_size))
                previous_state_image = numpy.copy(previous_state)
                previous_state_image -= previous_state_image.min()
                previous_state_image /= previous_state_image.max()
                previous_state_image *= 255
                cv2.imwrite('phong_state.png',previous_state_image)
                print(score)
