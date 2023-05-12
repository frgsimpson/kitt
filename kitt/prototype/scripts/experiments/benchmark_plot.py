#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarks nlpd plot

"""

import pandas as pd
import matplotlib.pyplot as plt
from kitt.config import PLOT_DIR, TABLES_DIR

plt.style.use('ggplot')
NLPD = True
Y_LABEL = 'NLPD' if NLPD else "RMSE"

filename = 'nlpd_uci' if NLPD else 'rmse_uci'
savefile = filename + '.txt'
df = pd.read_csv(TABLES_DIR / savefile, sep=' & ')
fig = plt.figure(figsize=(16, 4))
_ax = fig.add_subplot(111, frame_on=False)
_ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
colors = ['b', 'r', 'g', 'orange', 'steelblue', 'magenta']
model_cols = ['rbf_', 'as_', 'nkn_', 'ahgp_', 'kitt_', 'cl_']
models = ['RBF', 'Tree Search', 'NKN', 'AHGP', 'KITT (E+D)', 'KITT (E)']  # Must match above ordering
MARKERS = ['o', 's', 'd', '+', 'v', 'o', 's', 'd', '+', 'v']

n_models = len(model_cols)
for _ in range(8):  # iterate over subplots
    n_plot = _ + 1
    ax = fig.add_subplot(2, 4, n_plot)
    for m in range(n_models):
        mean = df[[col for col in df if col.startswith(model_cols[m])][0]][_]
        se = df[[col for col in df if col.startswith(model_cols[m])][1]][_]
        ax.errorbar(x=m, y=mean, yerr=se, c=colors[m], barsabove=True, capsize=4, label=models[m], marker=MARKERS[m],
                    markersize=10, markeredgecolor='k')
    ax.set_title(df['Dataset'][_], fontsize='small')
    if n_plot == 1 or n_plot == 5:
        ax.set_ylabel(Y_LABEL)

    plt.xticks([], [])

# Align y labels
fig.align_ylabels()
fig.tight_layout()

# get handles
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
# remove the errorbars
handles = [h[0] for h in handles]
print(handles)
print(labels)
plt.legend(handles=handles, labels=labels, loc='upper center',
           bbox_to_anchor=(-1.5, -0.1), fancybox=True, shadow=False, ncol=len(labels), framealpha=0., fontsize=16)

# SAVE
figfilename = filename + '.png'
plt.savefig(PLOT_DIR / figfilename, dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
