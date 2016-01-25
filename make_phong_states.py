import numpy,copy
import cv2
from phong_simulator import Simulator
import time
from tqdm import tqdm


def make_states(simulator,actor,epsilon,number_of_steps,number_of_games,winners_only=False):
    #initialize something to hold the games
    #assume epsilon decay happens outside this function
    game_list = []
    last_score_list = []
    for i in tqdm(range(number_of_games)):
        state_list = make_one_set(simulator,actor,epsilon,number_of_steps)
        last_score_list.append(state_list[-1][0][1]) #change this to average score
        game_list = game_list + state_list
    return game_list,numpy.mean(last_score_list)


def make_one_set(simulator,actor,epsilon,number_of_steps,display=False):
    state_list = []
    left_state_list = []
    previous_state = numpy.zeros((simulator.image_size,simulator.image_size,3))
    last_score = 0
    simulator.reset()
    previous_state[:,:,0] = numpy.reshape(simulator.screen,(simulator.image_size,simulator.image_size))
    #one set is one score ?
    for i in range(number_of_steps):
        state = []
        #check if the ball has crossed to the left field
        if simulator.ball_side() == "left":
            #flip the screen and play for the right side
            # screen = simulator.screen[:,::-1]
            #now do actions, e-greedy
            if numpy.random.uniform() < numpy.max([epsilon,0.1]):
                #do a random action
                left_action = numpy.random.randint(actor.number_of_actions)
            else:
                left_action = actor.return_action(previous_state[:,::-1])
            simulator.do_action(left_action,side="left")
        else:
            left_action = 0
        screen = simulator.screen
        #now do actions, e-greedy
        if numpy.random.uniform() < numpy.max([epsilon,0.1]):
            #do a random action
            action = numpy.random.randint(actor.number_of_actions)
        else:
            action = actor.return_action(previous_state)

        screen,score,points_made,end = simulator.do_action(action,side="right")
        if end == 0:
            #move the game along with a no-op!
            screen,score,points_made,end = simulator.do_action(0,side="right")

        if display:
            print('no-op            ', 'up          ','down')
            print(actor.display_output)
            cv2.imshow('Phong!',cv2.resize(screen,(0,0),fx=2,fy=2))
            cv2.waitKey(100)

        state.append([screen,score,action,points_made,previous_state])
        state_list.append(copy.deepcopy(state))

        state.append([screen[:,::-1],score,left_action,points_made,previous_state[:,::-1]])
        left_state_list.append(copy.deepcopy(state))

        previous_state[:,:,1:] = numpy.copy(previous_state[:,:,:2])
        previous_state[:,:,0] = numpy.reshape(screen,(simulator.image_size,simulator.image_size))

        if end != 0:
            #propogate the score backwards
            #figure out the direction of the score
            reward = simulator.reward*end #-1 if point on the actor, 1 otherwise
            discount_iterator = 0
            discount_factor = 0.85
            for previous_state in reversed(state_list):
                #this starts at the win, but doesn't add to the win reward
                #the reward can be decreased linearly or exponentially
                #this will do it linearly
                previous_state[0][1] += reward*numpy.power(discount_factor,discount_iterator)
                discount_iterator += 1
            #do the same, backwards, for left_action
            for previous_state in reversed(left_state_list):
                #this starts at the win, but doesn't add to the win reward
                #the reward can be decreased linearly or exponentially
                #this will do it linearly
                previous_state[0][1] += -1*reward*numpy.power(discount_factor,discount_iterator)
                discount_iterator += 1

            break

    return state_list #+ left_state_list

class FakeActor:
    def __init__(self,num_actions):
        self.number_of_actions = num_actions
        self.display_output = [0,0]
    def return_action(self,simulator_screen):
        return numpy.random.randint(self.number_of_actions)




if __name__ == "__main__":
    sim = Simulator(10)
    actor = FakeActor(3)
    start_time = time.time()
    game_state_list = make_states(sim,actor,1,200,10,winners_only=False)
    print(time.time() - start_time)
    for state_list in game_state_list:
        for state in state_list:
            print(state[0][1:4],numpy.mean(state[0][4]))
            cv2.imshow('Phong!',cv2.resize(state[0][4],(0,0),fx=2,fy=2))
            numpy.save('phong_screen',state[0][4])
            cv2.waitKey(400)
