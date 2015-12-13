from rl_model import ActionLearner
import theano
import theano.tensor as T
import numpy
from simulation_simulator import Simulator
from make_states import make_states
import random

if __name__=="__main__":
    rng = numpy.random.RandomState(1234)
    learning_rate = 0.01
    L2_reg = 0.0001
    #epsilon is the decision parameter - do you use the actor's actions or do them randomly?
    epsilon = 1
    epsilon_decay = 0.001
    sim = Simulator(128,10)
    x = T.tensor4('x')
    y = T.ivector('y')
    # screen = T.tensor4('screen')
    # action = T.vector()

    learner = ActionLearner(
        rng=rng,
        input=x,
        batch_size=50,
        n_filters=16,
        n_hidden=128,
        n_out=4
    )
    cost = (
        learner.single_action_cost(y)
        + L2_reg * learner.L2_sqr
    )
    gparams = [T.grad(cost, param) for param in learner.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(learner.params, gparams)
    ]


    learner.return_function( theano.function(
        inputs=[x],
        outputs=T.argmax(learner.output),
    )
    )
    train_model = theano.function(
        inputs=[x,y],
        outputs=cost,
        updates=updates,
    )

    for i in range(1000):
        current_epsilon =epsilon - epsilon_decay*i
        #create a batch of states
        state_list = make_states(sim,learner,current_epsilon,number_of_steps=10,number_of_games=100)
        #create a random selection of this state list for training
        states = random.sample(state_list,50)
        exit(0)
