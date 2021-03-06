from rl_tensorflow_model import ActionLearner
import numpy as np
from phong_simulator import Simulator
from make_phong_states import make_states,make_one_set,run_test_games
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
    parser.add_argument("--number_of_games",help="how many games to simulate per training run",default=200,type=int)
    parser.add_argument("--restore",help="to restore from the saved folder",default="No",type=str)
    parser.add_argument("--save_folder",help="where to save the session variables",default="/tmp/logs/",type=str)
    parser.add_argument("--learning_rate",help="how fast to learn",default=1e-5,type=float)
    parser.add_argument("--epsilon_decay",help="how quickly to use the actor in simulations (vs random actions)",default=1e-5,type=float)
    parser.add_argument("--display_iterations",help="how often to display a test game",default=10,type=int)
    parser.add_argument("--number_of_filters",help="how many filters the convolutional layer should have",default=16,type=int)
    parser.add_argument("--number_of_hidden",help="how many hidden units to have",default=256,type=int)
    parser.add_argument("--play_itself",help="whether this will play against itself or a simple Pong AI",default=0,type=int)
    parser.add_argument("--state_space",help="How many moves up and down the paddle can make",default=10,type=int)


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
    sim = Simulator(1,screen_size=args.image_size,state_space=args.state_space)
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
                image_size=sim.screen_size,
                n_filters=args.number_of_filters,
                n_hidden=args.number_of_hidden,
                n_out=sim.number_of_actions
                )
            learner.set_sess(sess)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            grads_and_vars = optimizer.compute_gradients(learner.single_action_cost) #could also use learner.normal_cost
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            loss_summary = tf.scalar_summary("cost", learner.single_action_cost)
            #visualize those first level filters
            filter_summary = tf.image_summary("filters",learner.first_level_filters,max_images=32)

            saver = tf.train.Saver(tf.all_variables())
            sess.run(tf.initialize_all_variables())
            summary_writer = tf.train.SummaryWriter(args.save_folder, sess.graph_def)
            if args.restore != "No":
                saver.restore(sess, args.save_folder+args.restore)

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  learner.x: x_batch,
                  learner.y: y_batch,
                  learner.dropout_keep_prob: 0.95
                }
                _, step,  loss,loss_summ,filter_summ = sess.run(
                    [train_op, global_step,  learner.single_action_cost ,loss_summary,filter_summary],
                    feed_dict)
                summary_writer.add_summary(loss_summ, step)
                time_str = datetime.datetime.now().isoformat()
                if step % 100 == 0:
                    summary_writer.add_summary(filter_summ, step)
                    print("{}: step {}, loss {}".format(time_str, step, loss))



            for i in range(20000):
                try:
                    current_step = tf.train.global_step(sess, global_step)
                    current_epsilon = np.max([0.1,epsilon - epsilon_decay*current_step]) #always have a little randomness

                    #create a batch of states
                    start_time = time.time()
                    state_list,avg_game_lengths = make_states(sim,learner,current_epsilon,number_of_steps=300,number_of_games=number_of_games,winners_only=False,play_itself=args.play_itself)
                    print('took {} seconds'.format(time.time() - start_time))
                    #create a random selection of this state list for training
                    screens = np.zeros((batch_size,sim.screen_size,sim.screen_size,3))
                    actions = np.zeros((batch_size,sim.number_of_actions),dtype=np.float32)
                    for j in range(int(len(state_list)/batch_size)):
                        #grab random batches from the training images
                        random_states = random.sample(state_list,batch_size)
                        index = 0
                        for state in random_states:
                            screens[index,:,:] = state[0][4]
                            actions[index,state[0][2]] = float(state[0][1])
                            index += 1
                        train_step(screens,actions)
                    if i % 5 == 0:
                        #save
                        print("saving at iteration {}, with epsilon of {}".format(i,current_epsilon))
                        saver.save(sess,args.save_folder+'model.ckpt', global_step=current_step)


                    if i % display_steps == 0 and current_step != 0 and i != 0:
                        #save
                        saver.save(sess,args.save_folder+'model.ckpt', global_step=current_step)
                        #do a test run
                        sim.reset()
                        for j in range(5):
                            game_score = run_test_games(sim,learner,number_of_steps=200,display=True,play_itself=0)
                            print("game score and length of games",game_score[-1],len(game_score))
                except KeyboardInterrupt:
                    #do a test run
                    sim.reset()
                    for j in range(4):
                        display_state_list = make_one_set(sim,learner,0,number_of_steps=100,display=True)
                        print("displaying against weak Pong AI")
                        display_state_list = make_one_set(sim,learner,0,number_of_steps=100,display=True,play_itself=0)
