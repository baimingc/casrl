import numpy as np
from loguru import logger
from scipy.special import loggamma


def update_alpha_posterior_dist(alpha_max, K, N, n_sample=100, alpha_min=0.01):
    alpha = np.linspace(alpha_min, alpha_max, n_sample)
    logp = (K - 1.5) * np.log(alpha) - 1.0 / (2.0 * alpha) + loggamma(alpha) - loggamma(N + alpha)
    alpha_select = alpha[np.argmax(logp)]
    return alpha_select

