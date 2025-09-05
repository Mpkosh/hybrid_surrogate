import matplotlib.pyplot as plt
import numpy as np


def plot(n_rows, n_cols, fontsize=14):
    fig, ax = plt.subplots(1, 1, figsize=)


x = np.linspace(0, 10, 20)

# single plot
fontsize = 14
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, 5*x, marker='o', color='OrangeRed', label='Linear dependency')
ax.plot(x, x**2, lw=3, color='RoyalBlue', label='Quadratic dependency')
ax.set_xlabel('X label here', fontsize=1.2*fontsize)
ax.set_ylabel('Y label here', fontsize=1.2*fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.legend(fontsize=1.2*fontsize)
ax.grid()
