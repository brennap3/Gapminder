# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 13:24:37 2016

@author: Peter
"""

 import sys

 sys.version ##grand job

 import theano
 
sys.hexversion

 x = theano.tensor.fvector('x')

 W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')


 import numpy

 W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')

 y = (x * W).sum()

 f = theano.function([x], y)

 output = f([1.0, 1.0])

 x = theano.tensor.fvector('x')

 ##In [10]: ## shared variable

 W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')

 f = theano.function([x], y,use on_unused_input='ignore')
 theano.function
 
import theano
import numpy
 
x = theano.tensor.fvector('x')
target = theano.tensor.fscalar('target')
 
W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')
y = (x * W).sum()
 
cost = theano.tensor.sqr(target - y)
gradients = theano.tensor.grad(cost, [W])
W_updated = W - (0.1 * gradients[0])
updates = [(W, W_updated)]
 
f = theano.function([x, target], y, updates=updates)
 

for i in xrange(10):
    output = f([1.0, 1.0], 20.0)
    print output
    
    



