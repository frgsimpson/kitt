""" Script to explore how to scale the priors in a product kernel.
Simplest case being a product of 2 or 3 rbf """

import numpy as np
import matplotlib.pyplot as plt

n_samples = 5_000
n_terms = 3  # Choose 2 or 3

# Draw random lengthscales in log space
log_l1 = np.random.randn(n_samples)
log_l2 = np.random.randn(n_samples)
log_l3 = np.random.randn(n_samples)

offset = np.log(n_terms)            # Shift the mean of the prior
scale = n_terms**0.25               # Scaling the standard deviation of the prior

log_l1_scaled = log_l1 * scale + offset
log_l2_scaled = log_l2 * scale + offset
log_l3_scaled = log_l3 * scale + offset

l1 = np.exp(log_l1)
l2 = np.exp(log_l2)
l3 = np.exp(log_l3)
l1_scaled = np.exp(log_l1_scaled)
l2_scaled = np.exp(log_l2_scaled)
l3_scaled = np.exp(log_l3_scaled)

# Compute effective lengthscales
if n_terms == 2:
    log_lprod = log_l1 + log_l2 - 0.5 * np.log(l1**2 + l2**2)
    log_lscaled = log_l1_scaled + log_l2_scaled - 0.5 * np.log(l1_scaled**2 + l2_scaled**2)
else:
    log_lprod = log_l1 + log_l2 + log_l3 - 0.5 * np.log(l1**2 + l2**2 + l3**2)
    log_lscaled = log_l1_scaled + log_l2_scaled + log_l3_scaled - 0.5 * np.log((l1_scaled * l2_scaled)**2 +
                                                                               (l1_scaled * l3_scaled)**2 +
                                                                               (l3_scaled * l2_scaled)**2)

print('Target mean, std:', np.mean(log_l1), np.std(log_l1))
print('Product mean, std:', np.mean(log_lprod), np.std(log_lprod))
print('Scaled mean, std:', np.mean(log_lscaled), np.std(log_lscaled))

bins = np.linspace(-3, 3, 64)

plt.hist(log_lprod, bins, alpha=0.5, label='Product')
plt.hist(log_l1, bins, alpha=0.5, label='l1')
plt.hist(log_l2, bins, alpha=0.5, label='l2')
plt.hist(log_lscaled, bins, alpha=0.5, label='Scaled')


plt.legend(loc='upper right')
plt.show()


