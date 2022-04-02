#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

## This script needs to be in the same directory as gaussian_kde.py for 
## this to work:
import gaussian_kde as gkde


## Create some random data, and manipulate so that it is located along a boundary:
rng = np.random.default_rng(2)
X = rng.standard_normal(2000)
Y = rng.standard_normal(2000)
X = X[X>0][0:900]
Y = Y[Y>0][0:900]

bounds = [[min(X), min(Y)], [max(X), max(Y)]]


#--- histogram ---------------------------------------------------------
xbins = np.linspace(min(X), max(X), 20)
ybins = np.linspace(min(Y), max(Y), 20)
xbinwidth, ybinwidth = xbins[1]-xbins[0], ybins[1]-ybins[0]
binarea = xbinwidth * ybinwidth
counts = np.histogram2d(X, Y, bins=[xbins, ybins], density=True)[0].T

hist = np.histogram(X, xbins, density=True)[0]


#--- KDEs --------------------------------------------------------------
xfine = np.linspace(-1, 4, 100)
yfine = np.linspace(-1, 4, 100)
binarea = (xfine[1]-xfine[0])*(yfine[1]-yfine[0])
xx, yy = np.meshgrid(xfine, yfine)
positions = np.vstack([xx.ravel(), yy.ravel()])
delta = 100

## KDE without boundary correction:
kern2D = gkde.GaussianKDE(np.vstack([X,Y]))
KDE2D  = np.reshape(kern2D.evaluate(positions), xx.shape)
print(kern2D.integrate_box(bounds, delta))

## KDE with truncated kernels:
kern2Dbt = gkde.GaussianKDE(np.vstack([X,Y]), bounds=bounds, weights=np.ones(len(X)))
KDE2Dbt  = np.reshape(kern2Dbt.evaluate(positions, bound_method='truncate'), xx.shape)
print(kern2Dbt.integrate_box(bounds, delta=50))

## KDE with boundary reflections:
kern2Dbr = gkde.GaussianKDE(np.vstack([X,Y]), bounds=bounds, weights=np.ones(len(X)))
KDE2Dbr  = np.reshape(kern2Dbr.evaluate(positions, bound_method='reflect'), xx.shape)
print(kern2Dbr.integrate_box(bounds, delta=50))


#--- plot --------------------------------------------------------------
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 7))
fig1.subplots_adjust(bottom=0.08, top=0.97, left=0.07, right=0.98)

for ax in [ax1, ax2, ax3, ax4]:
    rect = matplotlib.patches.Rectangle(
        (bounds[0][0], bounds[0][1]), width=bounds[1][0]-bounds[0][0], height=bounds[1][1]-bounds[0][1],
        facecolor='None', edgecolor='r', lw=2
    )
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.add_artist(rect)

ax1.set_ylabel(r'$y$ values')
ax3.set_ylabel(r'$y$ values')
ax3.set_xlabel(r'$x$ values')
ax4.set_xlabel(r'$x$ values')

ax1.text(3.5, -0.6, ha='right', s='2D histogram')
ax2.text(3.5, -0.6, ha='right', s='1D KDE')
ax2.text(3.5, -0.6, ha='right', color='white', s='2D KDE')
ax3.text(3.5, -0.6, ha='right', color='white', s='2D KDE with truncated kernels')
ax4.text(3.5, -0.6, ha='right', color='white', s='2D KDE with boundary reflection')

one   = ax1.pcolormesh(xbins, ybins, counts)
two   = ax2.pcolormesh(xfine, yfine, KDE2D)
three = ax3.pcolormesh(xfine, yfine, KDE2Dbt)
four  = ax4.pcolormesh(xfine, yfine, KDE2Dbr)

fig1.savefig('2D_example.png', dpi=170)
