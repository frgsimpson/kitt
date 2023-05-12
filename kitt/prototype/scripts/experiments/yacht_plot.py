import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
plt.style.use('ggplot')
MARKER_SIZE = 64

# Output from yacht_predict script
expressions = [['LINEAR*RBF'], ['LINEAR*MATERN52'], ['LINEAR*MATERN32'], ['MATERN52'], ['MATERN52*MATERN52'], ['RBF*COSINE'], ['RBF*MATERN32'], ['LINEAR*COSINE'], ['RBF'], ['RBF*MATERN52'], ['LINEAR*LINEAR'], ['MATERN32*MATERN52'], ['MATERN32*MATERN32'], ['MATERN32'], ['MATERN52*COSINE'], ['RBF*PERIODIC'], ['COSINE'], ['MATERN32*COSINE'], ['MATERN52*PERIODIC'], ['LINEAR*MATERN12'], ['MATERN32*PERIODIC'], ['PERIODIC*COSINE'], ['RBF*MATERN12'], ['MATERN12*MATERN52'], ['MATERN12*COSINE'], ['MATERN12*MATERN32'], ['MATERN12'], ['LINEAR*PERIODIC'], ['MATERN12*PERIODIC'], ['MATERN12*MATERN12'], ['LINEAR'], ['PERIODIC'], ['LINEAR*NOISE'], ['NOISE']]
kitt_scores = np.array([0.18206340681208533, 0.14827536714240783, 0.14555432040784796, 0.06672234801738018, 0.055524577365419756, 0.04697400999136091, 0.042380663373964074, 0.04098067455046293, 0.04056545756937219, 0.03998626492909444, 0.039114667650820466, 0.024268226731704545, 0.020779029844591707, 0.01995910527409789, 0.018151733713563975, 0.014612602680580285, 0.013100612736546738, 0.008909197823180897, 0.007616457177949078, 0.006663198590617106, 0.005489636608697289, 0.002646723568277941, 0.0024754563491047057, 0.0016947735701771786, 0.0015339800999008577, 0.001400336998824115, 0.0008843987785415471, 0.000810898427987906, 0.0005366862151054417, 0.00022766671101803266, 5.4128199388861386e-05, 3.9834218506575455e-05, 3.51040344895439e-06, 4.746797223947189e-08])
# kitt_scores = raw_kitt_scores[:-2] # omit start and end tokens
marg_likelihood = np.array([568.7,        659.38,             682.69,            512.59,          522.287,         493.8,           526.4,             -16.4,           496,       0.,             -51,                       525,                   492,                445,                504,            561,             -249,          487,            576,                 659,                     574,                   482,          0.,                  0.,               -1000,                   0.,                  292,                  602,                    0.,                  372,               -259,            503,        -324,           -399])
lpd = np.array([0.48787,         0.427,                0.695,            .2518,            .308,              0.32,           0.3689,           -2.4787,       -3.477,        0.,       -2.6685,                   0.26,                0.15,                -0.258,            0.1897,             -0.491,          -3.35,           0.06,        0.2933,                 -0.3805,                   -0.30444,            0.41,         0.,                  0.,                -290,                   0.,                  -1.11,                  0.37,          0.,                 -0.6883,           -3.477,          0.09,             -3.72     ,        -3.93  ])

# Make dict for easier access
# score_dict = dict()
# for i, expression in enumerate(expressions):
#     score_dict[expression] = kitt_scores[i]

valid_kitt_scores = kitt_scores[lpd != 0.]
valid_marg_likelihood = marg_likelihood[lpd != 0.]
valid_lpd = lpd[lpd != 0.]

# plt.scatter(valid_marg_likelihood, valid_lpd)
kitt_coeff = spearmanr(valid_kitt_scores, valid_lpd)
ml_coeff = spearmanr(valid_marg_likelihood, valid_lpd)

# bic_coeff = spearmanr(valid_bic, valid_lpd)

print('Kitt coeff:', kitt_coeff)
print('ML coeff:', ml_coeff)

fig = plt.figure(figsize=(12, 4))
plt.scatter(valid_kitt_scores, np.exp(valid_lpd), s=MARKER_SIZE, edgecolor='k')


# plt.xscale('log')
# plt.yscale('log')
xlims = [1e-3, 0.4]
log_ylims = [-0.6, 0.9]
ylims = np.exp(log_ylims)
# plt.xlim(xlims)
# plt.ylim(ylims)

plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)

plt.xlabel('KITT assigned probability')
plt.ylabel('Predictive density')

plt.savefig('yacht', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
