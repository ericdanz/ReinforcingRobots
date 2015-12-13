import numpy,copy
import cv2
from simulation_simulator import Simulator


def make_states(simulator,actor,epsilon,number_of_steps,number_of_games,winners_only=False):
    #initialize something to hold the games
    #assume epsilon decay happens outside this function
    game_list = []
    for i in range(number_of_games):
        simulator.reset(64,10)
        state_list = make_one_set(simulator,actor,epsilon,number_of_steps)
        if (len(state_list) < number_of_steps) or (not winners_only):
            for state in state_list:
                game_list.append(state)
    return game_list


def make_one_set(simulator,actor,epsilon,number_of_steps):
    state_list = []
    for i in range(number_of_steps):
        state = []
        if numpy.random.uniform() < epsilon:
            #do a random action
            action = numpy.random.randint(actor.number_of_actions)
        else:
            action = actor.return_action(simulator.screen)

        screen,score,end = simulator.do_action(action)
        #do the action another 3 times, just to move games along faster
        if end == None:
            screen,score,end = simulator.do_action(action)
        if end == None:
            screen,score,end = simulator.do_action(action)
        if end == None:
            screen,score,end = simulator.do_action(action)

        state.append([screen,score,action])
        state_list.append(copy.copy(state))

        #check if the actor won!
        if end:
            #propogate the reward back
            reward = score
            discount_iterator = 0
            discount_factor = 0.1
            for previous_state in reversed(state_list):
                #this starts at the state before the win
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
    sim = Simulator(64,10)
    actor = FakeActor(4)
    game_state_list = make_states(sim,actor,1,10,20,winners_only=False)
    for state_list in game_state_list:
        for state in state_list:
            print(state[1])
            cv2.imshow('sim',state[0])
            cv2.waitKey(1000)
