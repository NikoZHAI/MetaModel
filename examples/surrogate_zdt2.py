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
from neuro_surrogate import *
from benchmarks import zdt2, load_theo
import matplotlib.pyplot as plt
from matplotlib import rc


# Matplotlib
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# Load theoritical values
theo = load_theo('./ZDT/ZDT2.pf')


# ZDT2
PM_FUN = zdt2
DIMENSION = 30
POP_SIZE = 64
MAX_GENERATION = 25
MAX_EPISODE = 30
MUTATION_RATE = 0.08
MUTATION_U = 0.
MUTATION_ST = 0.2
REF=[1., 1.]


pop = Population(dim=DIMENSION, size=POP_SIZE, fitness_fun=PM_FUN,
                 max_generation=MAX_GENERATION)
pop.selection_fun = pop.compute_front
pop.mutation_fun = gaussian_mutator
pop.regions.append([0., 1.])
pop.crossover_fun = random_crossover
pop.mutaton_rate = MUTATION_RATE

# Parametrization
region = 0
params_ea = {'u': MUTATION_U,
             'st': MUTATION_ST,
             'trial_method': 'lhs',
             'trial_criterion': 'cm'}
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
     'alpha': 0.00001,
     'learning_rate': 'adaptive',
     'learning_rate_init': 0.002,
     'max_iter': 500,
     'verbose': False,
     'no_improvement_tol': 500,
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
        print("Episode: %s, Surrogate generation: %s, True front size: %s, "
              "Surrogate Front size: %s" %
              (i+1, pop.generation, pop.true_front[region].__len__(),
               pop.front[region].__len__()))
        pop.generation += 1

    # Re-evaluate the surrogate-sampled individuals using the PM
    pop.recalc_fitness_with(fun=PM_FUN, region=0)
    pop.select(region=region, **params_ea)
    pop.update_front(region=region, **params_ea)
    pop.update_true_front(region=region)
    s.fit(pop.render_features(region=region),
          pop.render_targets(region=region))
    pop.hypervol_metric(front=pop.true_front[region], ref=REF,
                    analytical=theo.as_matrix())
    pop.generation = 1

# ================================Visualization================================
fig, (ax, ax_metric) = plt.subplots(1, 2, figsize=(10,4), dpi=300)

# Test surrogate results
final_arc = pop.true_front[region]
x = []
y = []
for f in final_arc:
    x.append(f.fitness[0])
    y.append(f.fitness[1])


#eq1  = r'$\displaystyle f_1 = x_1$'
#eq2  = r'$\displaystyle f_2 = g(x)h(f_1(x),g(x))$'
#eq3 = r'$\displaystyle g(x)=1+\sum_{i=2}^{30} x_i$'
#eq4 = r'$\displaystyle h(f_1(x),g(x))=1-\left(\frac{f_1(x)}{g(x)}\right)^2$'
#eq5 = r'$x_i\in\left[0,1\right]$'
#ax.text(0., 0., eq1+'\n'+eq2+'\n'+eq3+'\n'+eq4+'\n'+eq5,
#        fontsize=12, size=12,
#        ha="left", va="bottom",
#        bbox=dict(boxstyle="square", fc=(1., 1., 1., 0.8), ec='#a5a5a5'))

ax.set_xlabel(r'$F_1$')
ax.set_ylabel(r'$F_2$', labelpad=0)
ax.set(title="Non-Dominated Front")

ax.set_ylim((0., 1.))
ax.set_xlim((0., 1.))

ax.scatter(theo.f1, theo.f2, c='orangered', s=1.5,
           label="Analytical (Zitzler et al. 2000)")
ax.scatter(x, y, c='royalblue', s=2.0, label="ANN Assisted NSGA-II")

# Plot legend.
lgnd = ax.legend(loc='lower left', numpoints=1)

# change the marker size manually for both lines
lgnd.legendHandles[0]._sizes = [10]
lgnd.legendHandles[1]._sizes = [10]
ax.grid(True, ls=':')

ax.text(0.9, 0.9, 'ZDT-2',
        ha='right', va='top', size=14,
        bbox={'boxstyle': 'round',
              'ec': (0, 0, 0),
              'fc': (1, 1, 1),
              }
        )


# ================================ Metrics ====================================
ax_hypervol = ax_metric.twinx()

# Calculate hypervolume coverage
hypervol_cov = [hv/pop.hypervol_ana for hv in pop.hypervol]

# Title
ax_metric.set_title('Hypervolume Metrics')

# Ranges
ax_metric.set_xlim((0, MAX_EPISODE))
ax_metric.set_ylim((0., 1.))
ax_hypervol.set_ylim((0., 1.))

# Lables
ax_metric.set_ylabel('Uncovered \ Hypervolume', color='royalblue', labelpad=-1)
ax_hypervol.set_ylabel('Hypervolume \ Coverage (\%)', color='orangered')
ax_metric.set_xlabel('Episode (N)')

# Ticks
ax_metric.set_yticklabels(ax_metric.get_yticks().round(1),
                          color='royalblue')
ax_hypervol.set_yticklabels(ax_hypervol.get_yticks().round(1),
                            color='orangered')

# Grid
ax_metric.grid(True, ls=':')

# Plot
metric = ax_metric.plot(pop.hypervol_diff, color='royalblue',
                        label='Uncovered Hypervolume', linewidth=2.0)
hypervol = ax_hypervol.plot(hypervol_cov,
                            label='Hypervolume Coverage (\%)',
                            color='orangered',
                            linewidth=2.0)

legends = metric + hypervol
labs = [l.get_label() for l in legends]

ax_metric.legend(legends, labs, loc='center right', fontsize=8)

plt.show()

