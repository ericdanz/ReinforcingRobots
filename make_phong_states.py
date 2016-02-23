import numpy as np
import copy
import cv2
from phong_simulator import Simulator
import time
from tqdm import tqdm


def make_states(simulator,actor,epsilon,number_of_steps,number_of_games,winners_only=False,play_itself=1):
    #initialize something to hold the games
    #assume epsilon decay happens outside this function
    game_list = []
    last_score_list = []
    for i in tqdm(range(number_of_games)):
        state_list = make_one_set(simulator,actor,epsilon,number_of_steps,play_itself=play_itself)
        last_score_list.append(state_list[-1][0][1]) #change this to average score
        game_list = game_list + state_list
    return game_list,np.mean(last_score_list)


def make_one_set(simulator,actor,epsilon,number_of_steps,display=False,play_itself=1):
    state_list = []
    left_state_list = []
    previous_state = np.zeros((simulator.screen_size,simulator.screen_size,3))
    last_score = 0
    action_numerator = 0
    per_action_penalty = 0
    simulator.reset()
    previous_state[:,:,0] = np.reshape(simulator.screen,(simulator.screen_size,simulator.screen_size))
    cv2.destroyAllWindows()
    #one set is one score ?
    for i in range(number_of_steps):
        state = []
        #check if the ball has crossed to the left field
        if simulator.ball_side() == "left" and play_itself:
            #flip the screen and play for the right side
            # screen = simulator.screen[:,::-1]
            #now do actions, e-greedy
            if np.random.uniform() < epsilon:
                #do a random action
                left_action = np.random.randint(actor.number_of_actions)
            else:
                left_action = actor.return_action(previous_state[:,::-1])
            simulator.do_action(left_action,side="left")
        elif simulator.ball_side() == "left":
            #do pong AI
            simulator.do_action(None,side="left",simple_AI=True)
            left_action = 0
        else:
            left_action = 0
        screen = simulator.screen
        #now do actions, e-greedy
        if np.random.uniform() < epsilon:
            #do a random action
            action = np.random.randint(actor.number_of_actions)
        else:
            action = actor.return_action(previous_state)

        screen,score,points_made,end = simulator.do_action(action,side="right")
        if action != 0:
            action_numerator += 1

        # if end == 0:
        #     #move the game along with a no-op!
        #     screen,score,points_made,end = simulator.do_action(0,side="right")

        if display:
            print('no-op            ', 'up          ','down')
            print(actor.display_output)
            if play_itself:
                display_title = "Phong RL vs RL"
            else:
                display_title = "Phong Simple vs RL"

            padded_screen = np.ones((screen.shape[0]+10,screen.shape[1]+10))
            padded_screen[5:screen.shape[0]+5,5:screen.shape[1]+5] = screen[:,:,0]
            cv2.imshow(display_title,cv2.resize(padded_screen,(0,0),fx=6,fy=6))
            cv2.moveWindow(display_title,10,10)
            cv2.waitKey(100)

        state.append([screen,score,action,points_made,previous_state])
        state_list.append(copy.deepcopy(state))

        state.append([screen[:,::-1],score,left_action,points_made,previous_state[:,::-1]])
        left_state_list.append(copy.deepcopy(state))

        previous_state[:,:,1:] = np.copy(previous_state[:,:,:2])
        previous_state[:,:,0] = np.reshape(screen,(simulator.screen_size,simulator.screen_size))

        if end != 0:
            #propogate the score backwards
            #figure out the direction of the score
            reward = simulator.reward*end #-1 if point on the actor, 1 otherwise
            discount_iterator = 0
            discount_factor = 0.99

            action_penalty = -.1
            if action_numerator !=0:
                per_action_penalty = action_penalty/action_numerator
            #actually, lets table the action penalty for now
            per_action_penalty = 0

            for previous_state in reversed(state_list):
                #this starts at the win, but doesn't add to the win reward
                #the reward can be decreased linearly or exponentially
                #this will do it linearly
                if previous_state[0][2] != 0:
                    previous_state[0][1] += per_action_penalty
                previous_state[0][1] += reward*np.power(discount_factor,discount_iterator)
                discount_iterator += 1
            # #do the same, backwards, for left_action
            # for previous_state in reversed(left_state_list):
            #     #this starts at the win, but doesn't add to the win reward
            #     #the reward can be decreased linearly or exponentially
            #     #this will do it linearly
            #     previous_state[0][1] += -1*reward*np.power(discount_factor,discount_iterator)
            #     discount_iterator += 1

            simulator.reset()
            break

    return state_list #+ left_state_list

def run_test_games(simulator,actor,number_of_steps,display=True,play_itself=0):
    state_list = []
    score_list = []
    left_state_list = []
    previous_state = np.zeros((simulator.screen_size,simulator.screen_size,3))
    last_score = 0
    action_numerator = 0
    per_action_penalty = 0
    epsilon=0
    simulator.reset()
    previous_state[:,:,0] = np.reshape(simulator.screen,(simulator.screen_size,simulator.screen_size))
    cv2.destroyAllWindows()
    #one set is one score ?
    for i in range(number_of_steps):
        state = []
        #check if the ball has crossed to the left field
        if simulator.ball_side() == "left" and play_itself:
            #flip the screen and play for the right side
            # screen = simulator.screen[:,::-1]
            #now do actions, e-greedy
            if np.random.uniform() < epsilon:
                #do a random action
                left_action = np.random.randint(actor.number_of_actions)
            else:
                left_action = actor.return_action(previous_state[:,::-1])
            simulator.do_action(left_action,side="left")
        elif simulator.ball_side() == "left":
            #do pong AI
            simulator.do_action(None,side="left",simple_AI=True)
            left_action = 0
        else:
            left_action = 0
        screen = simulator.screen
        #now do actions, e-greedy
        if np.random.uniform() < epsilon:
            #do a random action
            action = np.random.randint(actor.number_of_actions)
        else:
            action = actor.return_action(previous_state)

        screen,score,points_made,end = simulator.do_action(action,side="right")
        if action != 0:
            action_numerator += 1

        # if end == 0:
        #     #move the game along with a no-op!
        #     screen,score,points_made,end = simulator.do_action(0,side="right")

        if display:
            # print('no-op            ', 'up          ','down')
            # print(actor.display_output)
            if play_itself:
                display_title = "Phong RL vs RL"
            else:
                display_title = "Phong Simple vs RL"

            padded_screen = np.ones((screen.shape[0]+10,screen.shape[1]+10))
            padded_screen[5:screen.shape[0]+5,5:screen.shape[1]+5] = screen[:,:,0]
            cv2.imshow(display_title,cv2.resize(padded_screen,(0,0),fx=6,fy=6))
            cv2.moveWindow(display_title,10,10)
            cv2.waitKey(70)

        state.append([screen,score,action,points_made,previous_state])
        state_list.append(copy.deepcopy(state))
        score_list.append(copy.copy(score))

        # state.append([screen[:,::-1],score,left_action,points_made,previous_state[:,::-1]])
        # left_state_list.append(copy.deepcopy(state))

        previous_state[:,:,1:] = np.copy(previous_state[:,:,:2])
        previous_state[:,:,0] = np.reshape(screen,(simulator.screen_size,simulator.screen_size))

        # if end != 0:
        #     #propogate the score backwards
        #     #figure out the direction of the score
        #     reward = simulator.reward*end #-1 if point on the actor, 1 otherwise
        #     discount_iterator = 0
        #     discount_factor = 0.99
        #
        #     action_penalty = -.1
        #     if action_numerator !=0:
        #         per_action_penalty = action_penalty/action_numerator
        #     #actually, lets table the action penalty for now
        #     per_action_penalty = 0
        #
        #     for previous_state in reversed(state_list):
        #         #this starts at the win, but doesn't add to the win reward
        #         #the reward can be decreased linearly or exponentially
        #         #this will do it linearly
        #         if previous_state[0][2] != 0:
        #             previous_state[0][1] += per_action_penalty
        #         previous_state[0][1] += reward*np.power(discount_factor,discount_iterator)
        #         discount_iterator += 1
        #     # #do the same, backwards, for left_action
        #     # for previous_state in reversed(left_state_list):
        #     #     #this starts at the win, but doesn't add to the win reward
        #     #     #the reward can be decreased linearly or exponentially
        #     #     #this will do it linearly
        #     #     previous_state[0][1] += -1*reward*np.power(discount_factor,discount_iterator)
        #     #     discount_iterator += 1
        #
        #     break

    return score_list #+ left_state_list

class FakeActor:
    def __init__(self,num_actions):
        self.number_of_actions = num_actions
        self.display_output = [0,0]
    def return_action(self,simulator_screen):
        return np.random.randint(self.number_of_actions)




if __name__ == "__main__":
    sim = Simulator(10)
    actor = FakeActor(3)
    start_time = time.time()
    game_state_list = make_states(sim,actor,1,200,10,winners_only=False)
    print(time.time() - start_time)
    for state_list in game_state_list:
        for state in state_list:
            print(state[0][1:4],np.mean(state[0][4]))
            cv2.imshow('Phong!',cv2.resize(state[0][4],(0,0),fx=2,fy=2))
            np.save('phong_screen',state[0][4])
            cv2.waitKey(400)
