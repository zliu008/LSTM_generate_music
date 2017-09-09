#! /usr/bin/env python

from __future__ import print_function

import glob
import os
import sys

import numpy
try:
    import pylab
except ImportError:
    print ("pylab isn't available. If you use its functionality, it will crash.")
    print("It can be installed with 'pip install -q Pillow'")

from midi.utils import midiread, midiwrite
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#Don't use a python long as this don't work on 32 bits computers.
numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False


from lstm_generative_rev1 import *



def train(r, dt, files, f_grad_shared, f_grad_update, f_pred, \
batch_size=100, num_epochs=20, lr = 1E-6):

    assert len(files) > 0, 'Training set is empty!' \
                           ' (did you download the data files?)'
    dataset = [midiread(f, r,
                        dt).piano_roll.astype(theano.config.floatX)
               for f in files]

#    try:
    seq = None
    for epoch in range(num_epochs):
        numpy.random.shuffle(dataset)
        costs = []

        for s, sequence in enumerate(dataset):
            for i in range(0, len(sequence), batch_size):
                if i + 1 + batch_size < len(sequence):
                    loss = f_grad_shared(sequence[i:i + batch_size], sequence[i+1:i+batch_size+1])
                    costs.append(loss)
                    f_grad_update(lr)
                    if seq is None:
                        seq = sequence[i:i + batch_size]

        print('Epoch %i/%i' % (epoch + 1, num_epochs))
        print(numpy.mean(costs))
        sys.stdout.flush()

        numpy.savetxt('matrix.txt', seq, delimiter=',') 
    return seq
#
#    except KeyboardInterrupt:
#        print('Interrupted by user.')

def generate(initIn, r, dt, generate_function, filename, show=True):
    '''Generate a sample sequence, plot the resulting piano-roll and save
    it as a MIDI file.

    filename : string
        A MIDI file will be created at this location.
    show : boolean
        If True, a piano-roll of the generated sequence will be shown.'''
        
    piano_roll = numpy.round(generate_function(initIn))
    for repeat in range(1):
        piano_roll = numpy.concatenate((piano_roll, numpy.round(generate_function(piano_roll))), axis=0)
    midiwrite(filename, piano_roll, r, dt)
    if show:
        extent = (0, dt * len(piano_roll)) + r
        pylab.figure()
        pylab.imshow(piano_roll.T, origin='lower', aspect='auto',
                     interpolation='nearest', cmap=pylab.cm.gray_r,
                     extent=extent)
        pylab.xlabel('time (s)')
        pylab.ylabel('MIDI note number')
        pylab.title('generated piano-roll')


def test_lstm(batch_size=100, num_epochs=200):
    model_options = dict()
    model_options['dim_proj'] = 88
    model_options['encoder'] = 'lstm' #need to implement more options
    model_options['use_dropout'] = True
    params = init_params(model_options)
    tparams = init_tparams(params)
    (use_noise, x, y, f_pred, cost, 
     f_grad_shared, f_grad_update) = build_model(tparams, model_options)  
#    re = os.path.join(os.path.split(os.path.dirname(__file__))[0],
#                      'data', 'Nottingham', 'train', '*.mid')
    #re = 'C:/zaoliu/learningCases/NN_Python\\rnn_rbm\\data\\Nottingham\\train\\*.mid'
    re = os.path.join(os.getcwd(),
                      'data', 'Nottingham', 'train', '*.mid')
    ###Todo train function is no longer member function
    r=(21, 109)
    dt=0.3  ##need to supply files
    re = os.path.join(os.getcwd(),
                      'data', 'Nottingham', 'train', '*.mid')
    files = glob.glob(re)
    initX = train(r, dt, files, f_grad_shared, f_grad_update, f_pred, batch_size, num_epochs)
    generate(initX, r, dt, f_pred, 'sample_lstm_longer2.mid', show=True)
    return f_pred

if __name__ == '__main__':
    model_pred = test_lstm()
    ##Todo generate function is no longer member function
    ##model.generate('sample1.mid')
    #model.generate('sample2.mid')
    #pylab.show()    
    
###need to make sure the train function 
