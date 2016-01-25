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


UP_KEY = 63232
DOWN_KEY = 63233
LEFT_KEY = 63234
RIGHT_KEY = 63235
TURN_RIGHT = 101
TURN_LEFT = 113

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--image_size",help="size of simulation screen",default=64,type=int)
    parser.add_argument("-b","--batch_size",help="batch size for training",default=128,type=int)
    parser.add_argument("-g","--gpu",help="which gpu to use (-1 for cpu)",default=-1,type=int)
    parser.add_argument("--number_of_games",help="how many games to simulate per training run",default=200,type=int)
    parser.add_argument("--restore",help="to restore from the saved folder",default="No",type=str)
    parser.add_argument("--save_folder",help="where to save the session variables",default="/tmp/logs/",type=str)
    parser.add_argument("--learning_rate",help="how fast to learn",default=1e-4,type=int)
    parser.add_argument("--epsilon_decay",help="how quickly to use the actor in simulations (vs random actions)",default=0.001,type=float)
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

            saver = tf.train.Saver(tf.all_variables())
            sess.run(tf.initialize_all_variables())
            summary_writer = tf.train.SummaryWriter(args.save_folder, sess.graph_def)
            if args.restore != "No":
                saver.restore(sess, args.save_folder+args.restore)


            def redraw_heatmap(x,y,angle):
                #convert to radians
                angle = (angle - 90) *numpy.pi / 180.0
                #load the screen
                test_screen =numpy.random.randn(128,128,3)/100
                #other paddle location
                other_paddle_location = numpy.random.randint(128-12,size=(2))
                other_paddle_location[1] = int(128 *.1)
                test_screen[int(other_paddle_location[0]-12.0/2):int(other_paddle_location[0]+12.0/2),int(other_paddle_location[1]-1):int(other_paddle_location[1]+1),:] = 1
                #draw the balls
                yy, xx = numpy.mgrid[:128, :128]
                circle = (xx - x) ** 2 + (yy - y) ** 2
                test_screen[:,:,0] += (circle < 3**2)*2 #make the ball a little circle
                #put the other balls in order behind this ball
                circle = (xx - x+3*numpy.cos(angle)) ** 2 + (yy - y+3*numpy.sin(angle)) ** 2
                test_screen[:,:,1] += (circle < 3**2)*2 #make the ball a little circle
                #put the other balls in order behind this ball
                circle = (xx - x+6*numpy.cos(angle)) ** 2 + (yy - y+6*numpy.sin(angle)) ** 2
                test_screen[:,:,2] += (circle < 3**2)*2 #make the ball a little circle
                #do three columns, for each type of action

                noop_action_values = numpy.zeros(10)
                down_action_values = numpy.zeros(10)
                up_action_values = numpy.zeros(10)
                total_action_values = numpy.zeros(10) #these are the overall value of each position
                for i in range(10):
                    #move the paddle to the top
                    #they are 13 pixels high, and start at pixel 5
                    paddle = [ 5+13*i,18+13*i,114,116]
                    test_screen[:,113:,:] = -.11
                    test_screen[paddle[0]:paddle[1],paddle[2]:paddle[3],:] = 1
                    #find all the values of each state for the screen
                    _ = learner.return_action(test_screen)
                    action_values = learner.display_output[0]
                    print(action_values)
                    #0 is for i
                    total_action_values[i] += action_values[0]
                    noop_action_values[i] += action_values[0]
                    if i > 0:
                        total_action_values[i-1] += action_values[1]
                        up_action_values[i-1] += action_values[1]
                    if i < 9:
                        total_action_values[i+1] += action_values[2]
                        down_action_values[i+1] += action_values[2]
                #normalize the values
                total_action_values[0] /= 2 #the ends only get added to twice
                total_action_values[8] /= 2
                total_action_values[1:9] /= 3

                print(total_action_values)
                #make a heatmap of paddles
                total_action_values -= numpy.min(total_action_values)
                total_action_values /= numpy.max(total_action_values)

                noop_action_values -= numpy.min(noop_action_values)
                noop_action_values /= numpy.max(noop_action_values)

                up_action_values -= numpy.min(up_action_values)
                up_action_values /= numpy.max(up_action_values)

                down_action_values -= numpy.min(down_action_values)
                down_action_values /= numpy.max(down_action_values)

                print(numpy.min(total_action_values))
                print(numpy.max(total_action_values))
                print(total_action_values)
                test_screen[:,113:,:] = -.11
                ordered_total = numpy.argsort(total_action_values)
                for i in range(10):
                    paddle = [ 5+13*ordered_total[i],18+13*ordered_total[i],114,116]
                    test_screen[paddle[0]:paddle[1],paddle[2]:paddle[2]+2,:] = i*.1
                    paddle = [ 5+13*i,18+13*i,114,116]
                return test_screen

            #default starting point
            x = 60
            y = 60
            angle = 90
            key = -1
            test_screen = redraw_heatmap(x,y,angle)
            while True:
                if key != -1:
                    if key == UP_KEY:
                        y -= 2
                    elif key == DOWN_KEY:
                        y += 2
                    elif key == LEFT_KEY:
                        x -= 2
                    elif key == RIGHT_KEY:
                        x += 2
                    elif key == TURN_LEFT:
                        angle -= 10
                    elif key == TURN_RIGHT:
                        angle += 10

                    #move x y and angle
                    test_screen = redraw_heatmap(x,y,angle) #angle in degrees

                cv2.imshow('screen',cv2.resize(test_screen,(0,0),fx=4,fy=4 ))
                key = cv2.waitKey(1000)
            exit(0)
