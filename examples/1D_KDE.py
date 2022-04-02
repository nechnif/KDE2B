#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

## This script needs to be in the same directory as gaussian_kde.py for
## this to work:
import gaussian_kde as gkde


## Create some random data, and manipulate so that it is located along a boundary:
rng01 = np.random.default_rng(1001)
X = rng01.standard_normal(2000)
X = X[X>0][0:900]

bounds = [min(X), max(X)]

#--- histogram ---------------------------------------------------------
xbins = np.linspace(min(X), max(X), 15)
binwidth = np.diff(xbins)[0]
counts, bins = np.histogram(X, bins=xbins, density=True)

#--- KDEs --------------------------------------------------------------
xfine = np.linspace(-1, 4, 200)
finewidth = np.diff(xfine)[0]
delta=300

## KDE without boundary correction:
kern1D = gkde.GaussianKDE(X)
KDE1D  = kern1D.evaluate(xfine)
print(kern1D.integrate_box(bounds, delta=delta))

## KDE with truncated kernels:
kern1Dbt = gkde.GaussianKDE(X, bounds=bounds, weights=np.ones(len(X)))
KDE1Dbt  = kern1Dbt.evaluate(xfine, bound_method='truncate')
print(kern1Dbt.integrate_box(bounds, delta=delta))

## KDE with boundary reflections:
kern1Dbr = gkde.GaussianKDE(X, bounds=bounds, weights=np.ones(len(X)))
KDE1Dbr  = kern1Dbr.evaluate(xfine, bound_method='reflect')
print(kern1Dbr.integrate_box(bounds, delta=delta))


#--- plot --------------------------------------------------------------
fig1, ax1 = plt.subplots(1, 1, figsize=(7, 5))
fig1.subplots_adjust(bottom=0.12, top=0.97, left=0.1, right=0.96)

ax1.set_xlim(-0.5, 3.2)
ax1.set_xlabel(r'$x$ values')
ax1.set_ylabel('probability')

ax1.axvline(bounds[0], color='red', lw=2, label='boundary')
ax1.axvline(bounds[1], color='red', lw=2)

ax1.bar(bins[:-1], counts, width=binwidth, align='edge', alpha=0.4)
ax1.plot(xfine, KDE1D,   lw=2, ls='--', label='KDE')
ax1.plot(xfine, KDE1Dbr, lw=2, color='black', label='KDE with boundary reflection')
ax1.plot(xfine, KDE1Dbt, lw=2, color='orange', ls='-.', label='KDE with truncated kernels')

ax1.legend(loc='upper right')
fig1.savefig('1D_example.png', dpi=200)
