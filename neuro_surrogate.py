#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:22:07 2018

@author: niko
"""

import numpy as np
import copy as cp


def kursawe(x):
    f1 = np.multiply(-10.0,
                     np.exp(np.multiply(-0.2, np.sqrt( \
                        np.add(x[:-1].__pow__(2), x[1:].__pow__(2)))))).sum()
    f2 = np.add(np.abs(x).__pow__(0.8),
                np.sin(x.__pow__(3)).__mul__(5.0)).sum()

    return np.array([f1, f2])


def zdt1(x):
    f1 = x[0]
    g = np.add(1.0, np.multiply(9./29., x[1:].sum()))
    f2 = np.multiply(g, (1. - np.sqrt(np.divide(f1, g))))

    return np.array([f1, f2])


def zdt2(x):
    f1 = x[0]
    g = np.add(1.0, np.multiply(9./29., x[1:].sum()))
    f2 = np.multiply(g, (1. - np.square(np.divide(f1, g))))

    return np.array([f1, f2])


def zdt3(x):
    f1 = x[0]
    g = np.add(1.0, np.multiply(9./29., x[1:].sum()))
    f2 = np.multiply(g, (1. - np.sqrt(np.divide(f1, g)) - np.multiply(
                     np.divide(f1, g), np.sin(f1*10.*np.pi))))

    return np.array([f1, f2])


def zdt4(x):
    f1 = x[0]
    g = np.add(91., np.subtract(x[1:].__pow__(2.),
                                10. * np.cos(4. * np.pi * x[1:])).sum())
    f2 = np.multiply(g, (1. - np.sqrt(np.divide(f1, g))))

    return np.array([f1, f2])


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


def gaussian_mutator(gene, bounds, doomed, u=0., st=0.2,
                     multi_bounds=False, **kwargs):

    for i in range(gene.shape[0]):

        if not doomed[i]:
            continue
        else:
            b = bounds[i] if multi_bounds else bounds
            gene[i] += np.random.normal(u, np.multiply(np.abs(b).max(), st))

        if gene[i] < b[0]:
            gene[i] = b[0]
        if gene[i] > b[1]:
            gene[i] = b[1]

    return gene


class Individual(object):

    def __init__(self, dim=3, bounds=[-5., 5.], name='Undefined',
                 fitness=np.array([np.inf]), **kwargs):
        self.dimension = dim
        self._gene = 'vector of {} elements (parameters)...'.format(dim)
        self.name = name
        self._dominated = False
        self.bounds = np.add(bounds, [0., 1e-12])
        self.fitness = fitness
        self.cached = False
        self.multi_bounds = True

        if self.bounds.shape.__len__() == 1:
            self.gene = np.random.uniform(*self.bounds, dim)
            self.multi_bounds = False
        elif self.bounds.shape.__len__() == 2:
            self.gene = np.array([np.random.uniform(*b) for b in self.bounds])
        elif dim != self.bounds.shape[0]:
            raise NotImplementedError("Boundary and gene dimension does not "
                                      "match...")
        else:
            raise NotImplementedError("Problems with no boundary  or number "
                                      "of boundaries higher than 2 are not "
                                      "implemented yet...")

    def calc_fitness(self, fun):
        self.fitness = fun.__call__(self.gene)
        return self.fitness

    def mutate(self, routine, rate, **kwargs):
        doomed = np.random.sample(self.dimension).__lt__(rate)
        if doomed.any():
            self.gene = routine(self.gene, self.bounds, doomed,
                                multi_bounds=self.multi_bounds, **kwargs)
        return self.gene

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
        self.elites = []
        self.gene_len = 0
        self.regions = regions
        self.generation = 0
        self.front = []
        self.true_front = []
        self.cached = []
        self.max_generation = max_generation
        self.dim = dim

        self.fitness_fun = fitness_fun
        self.selection_fun = selection_fun
        self.mutation_fun = mutation_fun
        self.mutaton_rate = mutation_rate
        self.crossover_fun = crossover_fun

    def generate_init(self, region=0, **kwargs):

        local = [Individual(dim=self.dim, bounds=self.regions[region], \
                            **kwargs) for _i in range(self.size)]

        for i in local:
            i.calc_fitness(fun=self.fitness_fun)
            #self.cache(i)

        self.global_pop.append(local)
        self.elites.append([])
        self.front.append([])
        self.true_front.append([])

        self.gene_len = i.gene.shape[0]
        self.generation = 0
        return None

    def crossover(self, region=0, **kwargs):
        for i in self.global_pop.__getitem__(region):
            i.gene = self.crossover_fun(elites=self.front[region],
                                        gene_len=self.gene_len, **kwargs)
            i.mutate(self.mutation_fun, self.mutaton_rate)
            i.calc_fitness(fun=self.fitness_fun)
            #self.cache(i)
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
        for i in self.global_pop.__getitem__(region):
            i.gene = self.crossover_fun(elites=self.true_front[region],
                                        gene_len=self.gene_len, **kwargs)
            i.mutate(self.mutation_fun, self.mutaton_rate)
            i.calc_fitness(fun=self.fitness_fun)
            #self.cache(i)
        return None

