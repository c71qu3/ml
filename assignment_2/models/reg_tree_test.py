# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 19:53:10 2025

@author: mitre
"""

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from custom_regression_tree import CustomRegressionTree

rng = np.random.default_rng(0)

# X = rng.uniform(-1, 1, size=(200, 1))
# y = X[:, 0]**2 + rng.normal(0, 0.05, size=200)

n = 200

x1 = np.random.uniform(0, 1, n)
x2 = np.random.uniform(0, 1, n)

# nonlinear function
y = (
    3 * x1**2
    + 2 * (x2 > 0.5).astype(float)   
    + np.random.normal(0, 0.1, n)
)

X = np.column_stack([x1, x2])

y_min, y_max = np.min(y), np.max(y)
y_min = y_min - (y_min*0.1)
y_max = y_max + (y_max*0.1)

tree = CustomRegressionTree()
tree.fit(X, y)

preds = tree.predict(X[:5])

plt.figure()
plt.scatter(X[:, 0], y)
plt.scatter(X[:5, 0], preds)
plt.ylim(y_min, y_max)

plt.figure()
X_sorted = X #sorted(X)
pred_tree = tree.predict(X_sorted)
plt.scatter(X_sorted[:, 0], pred_tree)
plt.ylim(y_min, y_max)

tree.tree
