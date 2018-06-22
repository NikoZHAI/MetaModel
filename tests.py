#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 14:33:48 2018

@author: niko
"""

from neuro_surrogate import *
from multilayer_perceptron import MLPSurrogate
import numpy as np


params_surrogate = \
    {'hidden_layer_sizes': (8,),
     'activation': 'tanh',
     'solver': 'adam',
     'early_stopping': True,
     'batch_size': 8,
     'warm_start': True,
     'beta_1': 0.9,
     'beta_2': 0.999,
     'epsilon': 1e-12,
     'alpha': 0.001,
     'learning_rate': 'adaptive',
     'learning_rate_init': 0.002,
     'max_iter': 500,
     'verbose': True,
     'tol': 0.0001,
     }

X = np.random.uniform(-5., 5., size=(64*30, 3))
X = (X + 5.) / 10.
y0 = np.array([kursawe(x) for x in X])
y1 = np.array([kursawe(x)[:1] for x in X])
y2 = np.array([kursawe(x)[1:] for x in X])

m0 = MLPSurrogate(**params_surrogate)
m1 = MLPSurrogate(**params_surrogate)
m2 = MLPSurrogate(**params_surrogate)

m0.fit(X, y0)
m1.fit(X, y1.ravel())
m2.fit(X, y2.ravel())


for i in range(20):
    xx = (np.random.uniform(-5., 5., size=3) + 5.) / 10.
    yy = kursawe(xx)
    yy0 = m0.predict(xx.reshape(1, -1)).ravel()
    yy1 = m1.predict(xx.reshape(1, -1)).ravel()
    yy2 = m2.predict(xx.reshape(1, -1)).ravel()

    print('Calc: %s, m0: %s, m1: %s, m2: %s' % (yy,
            yy0-yy, yy1-yy[:1], yy2-yy[1:]))


