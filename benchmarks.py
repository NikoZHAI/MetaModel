#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 14:33:48 2018

@author: niko
"""

import numpy as np
import pandas as pd


def load_theo(path, n_objs=2):
    names = ['f'+str(i+1) for i in range(n_objs)]
    return pd.read_csv(filepath_or_buffer=path,
                       names=names,
                       delim_whitespace=True)




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


def zdt6(x):
    f1 = 1. - np.exp(-4. * x[0]) * np.power(np.sin(6.* np.pi * x[0]), 6)
    g = np.add(1.0, np.multiply(9.,
                                np.power(np.divide(x[1:].sum(), 9.), 0.25)))
    f2 = np.multiply(g, (1. - np.square(np.divide(f1, g))))

    return np.array([f1, f2])
