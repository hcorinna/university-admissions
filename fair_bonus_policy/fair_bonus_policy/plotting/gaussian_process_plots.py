#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:10:43 2020

@author: hertweck
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_gp(gp, X, y, points_alpha=0.1, with_gp_samples=False):
    plt.figure(figsize=(10, 8))

    difference = max(y) - min(y)
    ymin = min(y) - difference/10
    ymax = max(y) + difference/10
    xmin = min(X)[0]
    xmax = max(X)[0]
    X_ = np.linspace(xmin, xmax, xmax - xmin + 1)
    y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
    print('minimum bonus', X_[np.argmin(y_mean)])
    plt.plot(X_, y_mean, 'k', lw=3, zorder=9, label='Prediction')
    plt.fill_between(X_, y_mean - 1.9600 * y_std, y_mean + 1.9600 * y_std,
                     alpha=0.2, color='k', label='95% confidence interval')
    suggested_bonus = X_[np.argmin(y_mean)]
    plt.vlines(suggested_bonus, ymin, ymax, linestyles = '--', colors = 'g', label='Suggested policy: ' + str(suggested_bonus))
    
    if with_gp_samples:
        y_samples = gp.sample_y(X_[:, np.newaxis], 10)
        plt.plot(X_, y_samples, lw=1)
    if points_alpha > 0:
        plt.scatter(X[:, 0], y, c='blue', alpha=points_alpha, s=50, zorder=10, edgecolors=(0, 0, 0), label='Observations')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('Bonus value', size=22)
    plt.ylabel('Objective function', size=22)
    plt.title('Fit Gaussian process\n(%s)'
              % (gp.kernel_), size=24)

    print("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
              % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)))
    plt.tight_layout()
    plt.legend()

    plt.show()