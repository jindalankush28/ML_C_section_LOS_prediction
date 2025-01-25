from scipy.stats import binom
import numpy as np

# Given values
n = 5   # number of trials
p0 = 0.5  # null hypothesis probability
alpha = 0.05  # significance level

# Find the critical region
# We find the smallest value x for which P(X >= x | p0) <= alpha
p_values = binom.cdf(range(n+1), n, p0)
critical_region = np.where(p_values > 1 - alpha)[0][-1] + 1

# Part b: Checking the experiment outcome X = 4
X_observed = 4
reject_H0_experiment = X_observed >= critical_region