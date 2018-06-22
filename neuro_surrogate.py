#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:22:07 2018

@author: niko
"""

import numpy as np
import copy as cp
from operator import methodcaller
from pyDOE import lhs
from benchmarks import kursawe


def _rend_k_elites(pop, elites, **kwargs):
    for i in pop:
        inter = Individual(fitness=np.array([np.inf]*2))
        _i = 0
        dominated = None

        for e in elites:
            if i.__lt__(e) and inter.__lt__(e):
                inter = e
                dominated = _i
            _i += 1

        if dominated:
            elites.__setitem__(dominated, i)

    return elites


def random_crossover(elites, gene_len, **kwargs):
    child = [np.random.choice(elites).gene[i] \
             for i in range(gene_len)]
    return np.array(child)


def reject_acceptance(elites, gene_len, **kwargs):
    p = np.random.uniform()
    picked = np.random.choice(elites)
    rejected = []
    while (picked.acceptance != 1.) and (p > 0):
        if picked in rejected:
            picked = np.random.choice(elites)
            continue
        else:
            p -= picked.acceptance
            picked = np.random.choice(elites)
            rejected.append(picked)
    return picked


def calc_nsga_acceptance(pop, **kwargs):
    """ Calculate the NSGA-II acceptance of each individual in pop

        pop should be assigned by a pop_list[region] fashion.
    """
    if pop.__len__() < 5:
        [i.__setattr__('acceptance', 1.) for i in pop]
        return None

    total_dist = calc_cwd_dist(pop, **kwargs)

    for i in pop:
        i.acceptance = 1. if i._on_edge else np.divide(i._dist, total_dist)

    return None


def calc_cwd_dist(pop, **kwargs):
    """ Calculate the crowding distance of each individual in pop

        pop should be assigned by a pop_list[region] fashion.
    """
    if pop.__len__() <= 2:
        [i.__setattr__('cwd_dist', np.inf) for i in pop]
        return None

    for i in pop:
        i._dist = np.inf
        i._on_edge = False

    total_dist = 0.
    for j in range(pop[0].fitness.shape[0]):
        sort_by_fitness(tosort=pop, obj=j)
        pop[0]._on_edge = True
        pop[-1]._on_edge = True
        total_dist += np.sum([pop[i].update_dist(pop[i+1], pop[i-1], j) \
                              for i in range(1, pop.__len__()-1)])

    return total_dist


def nsga_crossover(elites, gene_len, **kwargs):
    child = [reject_acceptance(elites, gene_len, **kwargs).gene[i] \
             for i in range(gene_len)]
    return np.array(child)


def gaussian_mutator(gene, bounds, doomed, u=0., st=0.2, **kwargs):

    for i in range(gene.shape[0]):

        if not doomed[i]:
            continue
        else:
            b = bounds[i]
            gene[i] += np.random.normal(u, np.multiply(np.abs(b).max(), st))

        if gene[i] < b[0]:
            gene[i] = b[0]
        if gene[i] > b[1]:
            gene[i] = b[1]

    return gene


def sort_by_fitness(tosort, obj, reverse=False):
    tosort.sort(key=methodcaller('get_fitness', obj=obj), reverse=reverse)
    return None


def calc_hypervol(ref=None, front=None, minimize=True, **kwargs):
    """Calculate hypervolume metrics of a Pareto set, given a reference point.

    Parameters
    ----------
    ref : {1D-array-like}, shape (n_objectives, )
          The reference point.

    front : {array-like, sparse matrix}, shape (n_optimals, n_objectives)
            The Pareto optimals on which to calculate hypervolume.

    Returns
    -------
    hypevol : scalar, the calculated hypervolume between the reference point
              and the Pareto optimals.
    """
    hypevol = np.insert(front[:-1, 0], 0, ref[0]) - front[:, 0]

    if minimize:
        for i in range(1, front.shape[1]):
            hypevol *= (ref[i] - front[:, i])
    else:
        hypevol = -hypevol
        for i in range(1, front.shape[1]):
            hypevol *= (front[:, i] - ref[i])

    return hypevol.sum()


class Individual(object):

    def __init__(self, dim=3, bounds=[-5., 5.], name='Undefined',
                 fitness=np.array([np.inf]), trial_method='random',
                 gene=None, **kwargs):
        self.dimension = dim
        self._gene = 'vector of {} elements (parameters)...'.format(dim)
        self.name = name
        self._dominated = False
        self.bounds = np.add(bounds, [0., 1e-12])
        self.fitness = fitness
        self.cached = False
        self.acceptance = 0.
        self.trial_method = trial_method

        # Generate bounds
        if self.bounds.shape.__len__() == 1:
            self.bounds = np.repeat([self.bounds], self.dimension, axis=0)
        elif self.bounds.shape.__len__() == 2:
            if dim != self.bounds.shape[0]:
                raise NotImplementedError("Boundary and gene dimension "
                                          "does not match...")
        else:
            raise NotImplementedError("Problems with no boundary or "
                                      "number of boundaries higher than 2 "
                                      "are not implemented yet...")

        # Make trials
        if trial_method == 'random':
            self.gene = np.array([np.random.uniform(*b) for b in self.bounds])

        elif trial_method == 'lhs':
            self.gene = np.array([g * (b[1] - b[0]) + b[0] \
                                  for g, b in zip(gene, self.bounds)])
        else:
            raise NotImplementedError('%s trial design method is not '
                                      'implemented yet' % (self.trial_method))
        return None

    def calc_fitness(self, fun):
        self.fitness = fun.__call__(self.gene)
        return self.fitness

    def mutate(self, routine, rate, **kwargs):
        doomed = np.random.sample(self.dimension).__lt__(rate)
        if doomed.any():
            self.gene = routine(self.gene, self.bounds, doomed, **kwargs)
        return self.gene

    def get_fitness(self, obj):
        return self.fitness.__getitem__(obj)

    def update_dist(self, right, left, obj):
        self._dist += right.fitness[obj] - left.fitness[obj]
        return self._dist

    def __lt__(self, other):
        _better = self.fitness.__lt__(other.fitness)
        _crossed = self.fitness.__eq__(other.fitness)
        return _better.all() or (_crossed.any() and _better.any())

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return other.__lt__(self)

    def __ge__(self, other):
        return other.__lt__(self) or self.__eq__(other)

    def __eq__(self, other):
        return self.fitness.__eq__(other.fitness).all()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __ltarr__(self, other):
        return self.fitness.__lt__(other).all()

    def __learr__(self, other):
        return self.__ltarr__(other) or self.__eqarr__(other)

    def __gtarr__(self, other):
        return self.fitness.__gt__(other).all()

    def __gearr__(self, other):
        return self.__gtarr__(other).all() or self.__eqarr__(other)

    def __eqarr__(self, other):
        return self.fitness.__eq__(other).all()

    def __nearr__(self, other):
        return not self.__eqarr__(other)

    def __str__(self):
        return str(self.fitness) + " => " + str(self.gene)

    def save(self):
        return None


class Population(object):

    def __init__(self, dim=3, size=32,
                 fitness_fun=kursawe,
                 selection_fun=_rend_k_elites,
                 mutation_fun=None, mutation_rate=0.1,
                 crossover_fun=random_crossover,
                 regions=[], max_generation=200):
        self.global_pop = []
        self.size = size
        self.gene_len = 0
        self.regions = regions
        self.generation = 0
        self.verbose = False
        self.elites = []
        self.front = []
        self.true_front = []
        self.cached = []
        self.max_generation = max_generation
        self.dim = dim
        self.hypervol = []
        self.hypervol_diff = []

        self.fitness_fun = fitness_fun
        self.selection_fun = selection_fun
        self.mutation_fun = mutation_fun
        self.mutation_rate = mutation_rate
        self.crossover_fun = crossover_fun

        return None

    def generate_init(self, region=0, trial_method='random',
                      trial_criterion='cm', **kwargs):

        local = self.trial(method=trial_method, region=region,
                           criterion=trial_criterion)

        for i in local:
            i.calc_fitness(fun=self.fitness_fun)

        self.global_pop.append(local)
        self.elites.append([])
        self.front.append([])
        self.true_front.append([])

        self.gene_len = i.gene.shape[0]
        self.generation = 0

        return None

    def trial(self, method='random', region=0, **kwargs):
        if method == 'random':
            local = [Individual(dim=self.dim, bounds=self.regions[region], \
                     trial_method=method, **kwargs) for _i in range(self.size)]
        elif method == 'lhs':
            normalized_trials = self.trial_lhs(criterion='cm')
            local = [Individual(dim=self.dim, bounds=self.regions[region], \
                                trial_method=method, gene=g,
                                **kwargs) for g in normalized_trials]
        else:
            raise NotImplementedError('%s trial design method is not '
                                      'implemented yet' % (method))
        return local

    def trial_lhs(self, criterion='cm'):
        return lhs(n=self.dim, samples=self.size, criterion=criterion)

    def crossover(self, region=0, **kwargs):
        if 'nsga' in self.crossover_fun.__name__:
            calc_nsga_acceptance(pop=self.front[region], **kwargs)

        for i in self.global_pop.__getitem__(region):
            i.gene = self.crossover_fun(elites=self.front[region],
                                        gene_len=self.gene_len, **kwargs)
            i.mutate(self.mutation_fun, self.mutation_rate)
            i.calc_fitness(fun=self.fitness_fun)

        return None

    def select(self, region=0, rank=None, **kwargs):
        self.elites[region] = self.compute_front(region=region, **kwargs)
        return None

    def cache(self, individual, **kwargs):
        self.cached.append(individual)
        return None

    def evolve(self, region=0, **kwargs):
        self.generate_init(region=region, **kwargs)

        while self.generation <= self.max_generation:
            self.select(region=region, **kwargs)
            self.update_front(region=region, **kwargs)
            self.crossover(region=region, **kwargs)
            if self.verbose:
                print(self.generation)
            self.generation += 1
        return None

    def update_front(self, region=0, **kwargs):
        # Update current front
        updates = []
        if self.generation == 0:
            self.front[region] = self.elites[region].copy()
            return None
        else:
            current_front = self.front[region]

        for f in self.elites[region]:
            dominated = np.less_equal(current_front, f)
            if dominated.any():
                continue
            else:
                dominating = np.less(f, current_front)
                updates.append(cp.deepcopy(f))
                to_remove = np.where(dominating)[0]
                i_pop = 0
                for r in to_remove:
                    current_front.pop(r-i_pop)
                    i_pop += 1

        self.front[region].extend(updates)
        return None

    def compute_front(self, region=0, on=None, **kwargs):
        local = self.global_pop[region] if on is None else on
        front = []

        # Compute front of the current generation
        for i in local:
            if np.less(local, i).any() or np.equal(front, i).any():
                continue
            else:
                front.append(cp.deepcopy(i))
        return front

    def render_features(self, region=0):
        return np.array([i.gene for i in self.global_pop[region]])

    def render_targets(self, region=0):
        return np.array([i.fitness for i in self.global_pop[region]])

    def recalc_fitness_with(self, fun, region=0):
        # Re-calculate fitness in population
        for i in self.global_pop[region]:
            i.calc_fitness(fun)

        # Re-calculate fitness in current front
        for i in self.front[region]:
            i.calc_fitness(fun)

        # Re-evaluate the pareto front with PM results
        self.front[region] = self.compute_front(region=region,
                                                on=self.front[region])

        return None

    def update_true_front(self, region=0):
        """Use the surrogate front to update the current true Pareto front.
           The inaccuracy of the surrogate (overfitting or underfitting) may
           result in fake Pareto optimals which override the true ones. In
           order to save those true values, we build this so-called true_front
           archive to cache the PM's true Pareto optimal.
        """

        updates = []
        if self.generation == 0:
            self.true_front[region] = self.front[region].copy()
            return None
        else:
            current_front = self.true_front[region]

        for f in self.front[region]:
            dominated = np.less_equal(current_front, f)
            if dominated.any():
                continue
            else:
                dominating = np.less(f, current_front)
                updates.append(cp.deepcopy(f))
                to_remove = np.where(dominating)[0]
                i_pop = 0
                for r in to_remove:
                    current_front.pop(r-i_pop)
                    i_pop += 1

        self.true_front[region].extend(updates)
        return None

    def crossover_in_true_front(self, region=0, **kwargs):
        if 'nsga' in self.crossover_fun.__name__:
            calc_nsga_acceptance(pop=self.true_front[region], **kwargs)

        for i in self.global_pop.__getitem__(region):
            i.gene = self.crossover_fun(elites=self.true_front[region],
                                        gene_len=self.gene_len, **kwargs)
            i.mutate(self.mutation_fun, self.mutation_rate)
            i.calc_fitness(fun=self.fitness_fun)

        return None

    def find_least_crowded(self, region=0, _in='global', **kwargs):
        """ Find the least crowded solution in a set of non-dominated
            solutions (PM-evaluated by default)
        """
        if _in in ['global', 'pm', 'PM', 'Pm', 'true_front']:
            _front = self.true_front[region]
        elif _in in ['local', 'front']:
            _front = self.front[region]
        else:
            raise ValueError('Can not find the least crowded solution on "%"'
                             % _in)

        calc_cwd_dist(pop=_front, **kwargs)

        _init = True
        for i in _front:
            if i._on_edge:
                continue

            if _init:
                _least_crowded = i
                _init = False

            if _least_crowded._dist < i._dist: _least_crowded = i

            self._least_crowded = i

        return i

    def local_search(self, at, radius, pop_size= None,
                     max_generation=None, selection_fun=None,
                     mutation_fun=None, mutation_rate=None,
                     crossover_fun=None, trial_method='lhs',
                     trial_criterion='cm', u=0., st=0.2,
                     region=0, **kwargs):
        """ Apply MOEA local search at specific sub-regions

        """
        # Preparation
        dim = self.dim
        size= pop_size or self.size
        fitness_fun = self.fitness_fun
        selection_fun = selection_fun or self.selection_fun
        mutation_fun = mutation_fun or self.mutation_fun
        mutation_rate = mutation_rate or self.mutation_rate
        crossover_fun = crossover_fun or self.crossover_fun
        max_generation = max_generation or self.max_generation

        if hasattr(radius, '__iter__'):
            if not (radius.__len__() is self.gene_len):
                raise ValueError('If "radius" is an iterable object, its '
                                 'length (found %i) must be identical to the '
                                 'problem\'s dimentionality (found %i).'
                                 % (radius.__len__(), self.gene_len))
            elif radius.__class__.__name__ == 'ndarray':
                pass
            else:
                radius = np.array(radius)

        elif type(radius) not in [int, float]:
            raise TypeError('radius must be a digital number or a list of '
                            'digital numbers')
        else:
            pass

        if at.__class__.__name__ == 'Individual':
            _at = at.gene
        elif hasattr(at, '__iter__') and at.__len__() == self.gene_len:
            _at = at
        else:
            raise ValueError('"at" should either be paased an Individual or an'
                             ' array with the same length as an Individual\'s '
                             'gene.')

        # Generate local search population
        _pop = self._gen_local_search_pop(pioneer=at, r=radius, loc=_at,
                                          dim=dim, size=size,
                                          fitness_fun=fitness_fun,
                                          selection_fun=selection_fun,
                                          mutation_fun=mutation_fun,
                                          mutation_rate=mutation_rate,
                                          crossover_fun=crossover_fun,
                                          max_generation=max_generation)

        # Perform local search
        _pop.evolve(u=u, st=st, trial_method=trial_method,
                    trial_criterion=trial_criterion)

        return _pop.front

    def _gen_local_search_pop(self, pioneer, r, loc, dim, size, fitness_fun,
                              selection_fun, mutation_fun, mutation_rate,
                              crossover_fun, max_generation):

        # Generate local search bounds
        bounds = []
        _up_bounds = np.add(loc, r)
        _lo_bounds = np.subtract(loc, r)
        bounds.append([[_lb, _rb] for _lb, _rb in zip(_up_bounds, _lo_bounds)])
            # Restrict to the outter boundaries
        _i = 0
        for _ in np.greater(bounds[0], pioneer.bounds):
            if not _[0]: bounds[0][_i][0] = pioneer.bounds[_i][0]
            if _[1]: bounds[0][_i][1] = pioneer.bounds[_i][1]

        # Instantiate a new population
        _copy = Population(dim=dim, size=size, fitness_fun=fitness_fun,
                           selection_fun=selection_fun,
                           mutation_fun=mutation_fun,
                           mutation_rate=mutation_rate,
                           crossover_fun=crossover_fun,
                           max_generation=max_generation, regions=bounds)

        # Add the pioneer into the populations front
        _copy.front.append(pioneer)
        return _copy

#    def update_with_local_search(self):

    def hypervol_metric(self, front, ref, analytical=False, minimize=True):
        """ Calculate hypervolume metrics of the given front with an explicit
            reference point
        """
        # Sort the individuals in the front on one axis
        sort_by_fitness(tosort=front, obj=0, reverse=minimize)

        # Extract fitness from the Pareto optimals to form front_matrix
        front_matrix = np.array([[f for f in i.fitness] for i in front])

        # Calculate the current hypervolume given the reference
        self.hypervol.append(calc_hypervol(ref, front_matrix))

        if  analytical is False:
            self.hypervol_ana = 0.

        elif self.hypervol.__len__() == 1:
            # Sorting order of the analytical Pareto optimals, ascending if
            # maximization problem and vice versa
            order_ = -1 if minimize else 1

            # Sort the analytical Pareto Optimals
            a_ = analytical[analytical[:, 0].argsort()[::order_]]

            # Calculate the hypervolume between the reference and analyticals
            self.hypervol_ana = calc_hypervol(ref, a_)

        else:
            pass

        self.hypervol_diff.append(self.hypervol_ana - self.hypervol[-1])

        return None


