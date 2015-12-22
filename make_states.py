import numpy,copy
import cv2
from simple_scrolling_simulator import Simulator
import time


def make_states(simulator,actor,epsilon,number_of_steps,number_of_games,winners_only=False):
    #initialize something to hold the games
    #assume epsilon decay happens outside this function
    game_list = []
    length_list = []
    for i in range(number_of_games):
        state_list = make_one_set(simulator,actor,epsilon,number_of_steps)
        #tying the mix of wins to epsilon allows a decay in random games, leading towards
        #more wins as the learning progresses
        while (len(state_list) == number_of_steps) and (winners_only) and (numpy.random.uniform() > (numpy.min([0,epsilon]) + 0.01) ):
            state_list = make_one_set(simulator,actor,epsilon,number_of_steps)
        length_list.append(len(state_list))
        game_list = game_list + state_list
    print("The average game length (lower is better, and 10 is the max): {}".format(numpy.mean(length_list)))
    return game_list,numpy.mean(length_list)


def make_one_set(simulator,actor,epsilon,number_of_steps,display=False):
    state_list = []
    simulator.reset(simulator.image_size,10)
    for i in range(number_of_steps):
        state = []
        if numpy.random.uniform() < epsilon:
            #do a random action
            action = numpy.random.randint(actor.number_of_actions)
        else:
            action = actor.return_action(simulator.screen)

        screen,score,end = simulator.do_action(action)
        if display:
            print('left           ','right           ','up          ','down')
            print(actor.display_output)
            cv2.imshow('sim',cv2.resize(screen,(0,0),fx=2,fy=2))
            cv2.waitKey(100)
        state.append([screen,score,action])
        state_list.append(copy.copy(state))

        #check if the actor won!
        if end:
            #propogate the reward back
            reward = score
            discount_iterator = 0
            discount_factor = 0.1
            for previous_state in reversed(state_list):
                #this starts at the win, but doesn't add to the win reward
                #the reward can be decreased linearly or exponentially
                #this will do it linearly
                previous_state[0][1] += reward*(1- (discount_factor*discount_iterator))
                discount_iterator += 1

            break

    return state_list

class FakeActor:
    def __init__(self,num_actions):
        self.number_of_actions = num_actions

    def return_action(self,simulator_screen):
        return numpy.random.randint(self.number_of_actions)


if __name__ == "__main__":
    sim = Simulator(128,10)
    actor = FakeActor(2)
    start_time = time.time()
    game_state_list = make_states(sim,actor,1,10,100,winners_only=False)
    print(time.time() - start_time)
    for state_list in game_state_list:
        for state in state_list:
            print(state[0][1])
            cv2.imshow('sim',state[0][0])
            cv2.waitKey(1000)
