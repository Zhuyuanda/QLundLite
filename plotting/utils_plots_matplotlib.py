import numpy as np
import matplotlib.pyplot as plt

def hist_with_errors(data, bins=10, range=None, density=None, weights=None, **kwargs):
    """
    Create a histogram with error bars representing the statistical uncertainty in each bin, assuming a Poisson distribution where the error is the square root of the count in each bin.
    The arguments are the same as for numpy.histogram.
    Any additional keyword arguments are passed to matplotlib.pyplot.errorbar.
    """
    hist_args = dict(bins=bins, range=range, density=False)

    counts, bin_edges = np.histogram(data, **hist_args, weights=weights)
    midpoints = (bin_edges[:-1] + bin_edges[1:])*0.5

    if weights is None:
        errors = np.sqrt(counts)
    else:
        errors = np.sqrt(np.histogram(data, **hist_args, weights=np.array(weights)**2)[0])

    if density:
        norm_fact = 1/np.sum(np.diff(bin_edges)*counts)
        counts = counts*norm_fact
        errors = errors*norm_fact

    plt.errorbar(midpoints, counts, yerr=errors, **kwargs)