import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
rc('text', usetex=True)

def plot_betatrace(beta):
    nbeta,M,P = beta.shape

    fig,ax = plt.subplots( ncols=M, sharey='row' )
    for m in range(M):
        for p in range(P):
            if not all(np.array([beta[i,m,p] for i in range(nbeta) for m in range(M)]) == 0):
                ax[m].plot( beta[:,m,p], label=r'$\beta_{%i,%i}$' % (m+1,p) )
                ax[m].scatter( range(nbeta), beta[:,m,p], marker='.' )
        ax[m].legend()

    return(fig, ax)
