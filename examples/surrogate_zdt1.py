#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:16:30 2018

@author: niko

"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from multilayer_perceptron import MLPSurrogate
# from multioutput import MultiOutputRegressor
from neuro_surrogate import Population, gaussian_mutator, zdt1,\
                            random_crossover, Individual
import matplotlib.pyplot as plt
from matplotlib import rc


# Matplotlib
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


PM_FUN = zdt1
DIMENSION = 30
POP_SIZE = 128
MAX_GENERATION = 25
MAX_EPISODE = 30
MUTATION_RATE = 0.06
MUTATION_U = 0.
MUTATION_ST = 0.2


pop = Population(dim=DIMENSION, size=POP_SIZE, fitness_fun=PM_FUN,
                 max_generation=MAX_GENERATION)
pop.selection_fun = pop.compute_front
pop.mutation_fun = gaussian_mutator
pop.regions.append([0., 1.])
pop.crossover_fun = random_crossover
pop.mutaton_rate = MUTATION_RATE

# Parametrization
region = 0
params_ea = {'u': MUTATION_U, 'st': MUTATION_ST}
params_surrogate = \
    {'hidden_layer_sizes': (6, 8),
     'activation': 'tanh',
     'solver': 'adam',
     'early_stopping': False,
     'batch_size': 4,
     'warm_start': True,
     'beta_1': 0.9,
     'beta_2': 0.999,
     'epsilon': 1e-12,
     'alpha': 0.001,
     'learning_rate': 'adaptive',
     'learning_rate_init': 0.002,
     'max_iter': 500,
     'verbose': True,
     }

# ===============================Initialization================================

# Generation of first population
pop.generate_init(region=region, **params_ea)
pop.select(region=region, **params_ea)
pop.update_front(region=region, **params_ea)
pop.update_true_front(region=region)

# Initialize and train the surrogate
s = MLPSurrogate(**params_surrogate)
s.fit(pop.render_features(region=region),
      pop.render_targets(region=region))

# Replace the PM fitness function by the surrogate
pop.fitness_fun = s.render_fitness
pop.crossover_in_true_front(region=region, **params_ea)

# ===============================Meta Modelling================================

for i in range(1, MAX_EPISODE):
    # Evolutional computation on the surrogate
    while pop.generation <= pop.max_generation:
        pop.select(region=region, **params_ea)
        pop.update_front(region=region, **params_ea)
        pop.crossover_in_true_front(region=region, **params_ea)
        print("Episode: %s, Surrogate generation: %s, True front size: %s" %
              (i+1, pop.generation, pop.true_front[region].__len__()))
        pop.generation += 1

    # Re-evaluate the surrogate-sampled individuals using the PM
    pop.recalc_fitness_with(fun=PM_FUN, region=0)
    pop.select(region=region, **params_ea)
    pop.update_front(region=region, **params_ea)
    pop.update_true_front(region=region)
    s.fit(pop.render_features(region=region),
          pop.render_targets(region=region))
    pop.generation = 1

# ================================Visualization================================

# Test surrogate results
final_arc = pop.true_front[region]
x = []
y = []
print('Best Solutions: \n')
for f in final_arc:
    x.append(f.fitness[0])
    y.append(f.fitness[1])
    # print(f)


def load_theo():
    return pd.read_csv(filepath_or_buffer='./ZDT/ZDT1.pf', names=['f1', 'f2'],
                       delim_whitespace=True)

theo = load_theo()
fig, ax = plt.subplots(figsize=(8,6), dpi=100)
ax.scatter(theo.f1, theo.f2, c='orangered', s=1.2,
           label="Analytical (F. Kursawe 1991)")
ax.scatter(x, y, c='royalblue', s=1.6, label="Surrogate NSGA-II")
ax.set_xlabel(r'$\displaystyle f_1=\sum_{i=1}^2'
              r'\left[-10exp\left(-0.2\sqrt{(x_i^2+x_{i+1}^2)}\right)\right]$')
ax.set_ylabel(r'$\displaystyle f_2=\sum_{i=1}^3'
              r'\left[|x_i|^{0.8}+5sin(x_i^3)\right]$')
ax.set(title="Kursawe")

# Plot legend.
lgnd = ax.legend(numpoints=1)

# change the marker size manually for both lines
lgnd.legendHandles[0]._sizes = [10]
lgnd.legendHandles[1]._sizes = [10]
plt.grid()
plt.show()
fig.savefig('zdt1_tanh.png', format='png')


for i in range(50):
    ind = Individual(dim=30, bounds=[0., 1.])
    diff = np.subtract(pop.fitness_fun(ind.gene), PM_FUN(ind.gene))
    print(diff)


