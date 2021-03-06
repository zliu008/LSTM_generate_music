from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb

datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.3, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.3)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
   
    #this will get param_init_lstm
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    #W contains Wi, Wf, Wc, Wo for input, forget, state and output units
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm'):
    nsteps = state_below.shape[0]
#    if state_below.ndim == 3:
#        n_samples = state_below.shape[1]
#    else:
#        n_samples = 1

    #assert mask is not None #make sure mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        if _x.ndim == 2:
            return _x[:, n * dim:(n + 1) * dim]
        return _x[n * dim:(n + 1) * dim]

    def _step(x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
	#whenever a mask = 0 , the input is not a valid word, thus the state update should also be masked (use the previous state)
      #  c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
       # h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c
    #state_below = input
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])
    #replace state_below with state_below = input * W + b
    #state_below has the shape of (97, 16, 128*4), in which 4 represent 4 W matrices.
    # state_below.shape[0] represent the time steps, which is also the length of the longest sentence.
    # The LSTM is recurrent word by word through the 16 sentences paralllelly.
    # the two outputs in rval = (h, c) is feedback to _step function as inputs to calcuate next state. 
    # since LSTM state can have long term memory, it could be used to handle long sentence, for example sentence with 96 words.   
    dim_proj = options['dim_proj']
#    rval = _step(state_below[0,:], 
#                 tensor.alloc(numpy_floatX(0.),
#                              n_samples,
#                              dim_proj),
#                 tensor.alloc(numpy_floatX(0.),
#                               n_samples,
#                               dim_proj))
                               
                              
    rval, updates = theano.scan(_step,
                                sequences=[state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           #n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           #n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    # the gradient in gshared is updated by specifying updates = gsup
    # each time f_grad_shared is called, it caculate the gradient and store it in gshared
    # which will be used for parameter update in f_update
    # I think this two-step fashion is just to be compatible with other optimizers used in this script. 
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    #todo, TypeError: ('An update must have the same type as the original shared variable (shared_var=lstm_U_grad, shared_var.type=TensorType(float64, matrix), update_val=DimShuffle{1}.0, update_val.type=TensorType(float64, vector)).',
    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)] #careful that udn is a pair(,) since it is from updir_new, and only the [1] term is the updated value
    #parameters update with p = p + udn[1], and ud is also updating
    #with ud = 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    #x = tensor.matrix('x', dtype='int64')
    #y = tensor.vector('y', dtype='int64')
    x = tensor.matrix('x', dtype = 'float64')
    y = tensor.matrix('y', dtype='float64')
#    n_timesteps = x.shape[0]
#   
#    #the shape of x is (97, 16) tparams['Wemb'] is (10000, 128) where 10000 is n_words.
#    #The matrix tparams['Wemb'] used a 128-length vector to uniquely encode the 10000 words.
#    #The LSMT in this design also happens to have 128 hidden units connecting to the softmax function.
#    #This means that each W, U should be (128, 128), and b be (128,).
#    #However, since there are 4 sets of W, U, b for i, c, f, o, this script uses 
#    #lstm_W, lstm_U, lstm_b to store all the W, U, b matrices. Thus, 
#    #lstm_W is (128, 512), lstm_U is (128, 512), lstm_b is (512, ). 
#    #but this does not mean we need to use 512 hidden units. Remember that the output 'o'
#    #to softmax is always 128. The lumped lstm_W, lstm_U, and lstm_b is just for convienience
#    #in computation. In this way, we can compute the result in just 1 matrix multiplication(and addition)
#    #step, and then slicing the results into i, c, f, and o. See _step() function for detais. 
#    
#    #x is a list of sentences, and for example, is in shape of (97, 16). 16 is the nunmber
#    #of the sentences, and 97 is the length of the longest sentence. 
#    #We want to encode the words in these sentences with Wemb. What we do is to 
#    #map each word into one of the 10000 col vectors in Wemb as the following. 
#    #Use x.flattern to index the 97*16 col vectors (each with length 128) from tparams['Wemb'] 
#    #and reshape the (97*16, 128) matrix into (97, 16, 128). 
#    #once the code goes to get_layer(options['encoder'])[1], it will goto lstm_layer (see 'get_layer function' and 'layers' definition),
#    #in which emb will be used as the initial input of state_below to calculate
#    #state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
#    #               tparams[_p(prefix, 'b')])
#    #here tparams[_p(prefix, 'W') is (128,512), and thus the result of 
#    #the dot product is (97,16, 512). tparams['lstm_b'] is (512, ) the '+' operation
#    #add the bias b through broadcasting since 512 in (512,) matches the last dimenstion of (97, 16, 512)
##    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
##                                                #n_samples,
##                                                options['dim_proj']])
##                                                
##    Yemb = tparams['Wemb'][y.flatten()].reshape([n_timesteps,
##                                                #n_samples,
##                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, x, options,
                                            prefix=options['encoder']
                                            )
    
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
    
    #since this is a classification problem, U is (128, 2) and b is (2,)
    #pred below will store the probabilities for the two classes 0, 1 
    #since proj is (16, 128), pred is (16, 2)
    #pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    pred_vec = proj#[:,0,:]
    pred =   tensor.nnet.sigmoid(pred_vec)
    
    #pred = tensor.argmin( tensor.sum(( tparams['Wemb'] - pred_vec) ** 2))
   
    #f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    #f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')
    f_pred = theano.function([x], pred, name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    #cost = tensor.sum(tensor.nnet.categorical_crossentropy(pred, Yemb)) + off
    #cost = tensor.sum( (pred - y) ** 2) / pred.shape[0]
    L =  -theano.tensor.sum( y * theano.tensor.log(pred) \
    + ( 1 - y ) * theano.tensor.log(1 - pred), axis = 0)/pred.shape[0]
    cost = theano.tensor.mean(L)
    
    grads = tensor.grad(cost, wrt=list(tparams.values()))
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = adadelta(lr, tparams, grads,
                                        x, y, cost)

    return use_noise, x, y, f_pred, cost, f_grad_shared, f_update


def pred_error(f_pred,  X_data, y_data):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    y_pred= f_pred(X_data)
    accuracy = (y_pred == y_data ).sum()
    valid_err = 1 - numpy.float(accuracy) / len(y_pred)
    return valid_err

