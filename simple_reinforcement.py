from rl_model import ActionLearner



if __name__=="__main__":
    rng = numpy.random.RandomState(1234)
    learning_rate = 0.01
    L2_reg = 0.0001

    x = T.matrix('x')
    y = T.ivector('y')

    actor = ActionLearner(
        rng=rng,
        input=x,
        batch_size=50,
        n_filters=16,
        n_hidden=128,
        n_out=4):
    )
    cost = (
        actor.single_action_cost(y)
        + L2_reg * actor.L2_sqr
    )
    gparams = [T.grad(cost, param) for param in actor.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(actor.params, gparams)
    ]

    test_model = theano.function(
        inputs=[screen],
        outputs=actor.return_action(),
        givens={
            x: screen
        }
    )
    train_model = theano.function(
        inputs=[state],
        outputs=cost,
        updates=updates,
        givens={
            x: state[0],
            y: state[1]
        }
    )
