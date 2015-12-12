import numpy,copy
import cv2
from simulation_simulator import Simulator



def make_states(simulator,actor,epsilon,number_of_steps,number_of_games):
    #initialize something to hold the games
    #assume epsilon decay happens outside this function
    game_list = []
    for i in range(number_of_games):
        state_list = make_one_set(simulator,actor,epsilon,number_of_steps)
        game_list.append(state_list)
    return game_list



def make_one_set(simulator,actor,epsilon,number_of_steps):
    state_list = []
    for i in range(number_of_steps):
        state = []
        if numpy.random.uniform() < epsilon:
            #do a random action
            action = numpy.random.randint(actor.number_of_actions)
            screen,score = simulator.do_action(action)
            screen,score = simulator.do_action(action)
            screen,score = simulator.do_action(action)
            state.append([screen,score,action])
            state_list.append(copy.copy(state))
        else:
            action = actor.return_action(simulator.screen)
            screen,score = simulator.do_action(action)
            screen,score = simulator.do_action(action)
            screen,score = simulator.do_action(action)
            state.append([screen,score,action])
            state_list.append(copy.copy(state))
    return state_list

class FakeActor:
    def __init__(self,num_actions):
        self.number_of_actions = num_actions

    def return_action(self,simulator_screen):
        return numpy.random.randint(self.number_of_actions)



if __name__ == "__main__":
    sim = Simulator(256,10)
    actor = FakeActor(4)
    game_state_list = make_states(sim,actor,1,10,10)
    for state_list in game_state_list:
        for state in state_list:
            cv2.imshow('sim',state[0][0])
            cv2.waitKey(1000)
