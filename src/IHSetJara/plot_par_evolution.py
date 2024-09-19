import matplotlib.pyplot as plt

def plot_par_evolution(results):
    """
    Plot the evolution of a parameter over the generations.
    
    Parameters
    ----------
    results : numpy structured array
        The results of the optimization.
    par : str
        The name of the parameter to plot.
    """
    labels = [word for word in results.dtype.names if word.startswith('par')]

    n_par = len(labels)

    fig, ax = plt.subplots(n_par, 1, figsize=(8, 2*n_par), dpi=200, linewidth=5, edgecolor="#04253a")
    for i, par in enumerate(labels):
        ax[i].plot(results[par], color='black', linestyle='solid', linewidth=.75)
        ax[i].set_ylabel(par[3:])
        ax[i].set_xlabel('Generation')
    plt.show()
