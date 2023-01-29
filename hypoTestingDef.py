import os, sys, pathlib, time, re, glob, math, copy
from datetime import datetime
import warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import optimize
from scipy import stats
from tqdm import tqdm

##########################################################################################
def main():
    verbosity = 1
    rangeX = [-7, 22]
    plotRes = 1000
    muSigNull = [0.0, 1.8]
    muSigAlt  = [5.9, 2.5]
    dataLoc = 4.5

    qVal     = np.linspace(*rangeX, plotRes)
    baseline = qVal*0.0
    gausNull = gaussian(*muSigNull, qVal)
    gausAlt  = gaussian(*muSigAlt,  qVal)

    qAlpha = muSigNull[0]+2*muSigNull[1]    #2 sigma-significance
    qVal_CRsel = (qVal > qAlpha)            #CR for critical region
    qVal_PVsel = (qVal > dataLoc)           #PV for p-value
    qVal_PSsel = (qVal > dataLoc)           #PS for power/sensitivity
#plots
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    
    nullDist = ax0.plot(qVal, gausNull, linewidth=2, alpha=0.8, color="blue")[0]
    altDist  = ax0.plot(qVal, gausAlt,  linewidth=2, alpha=0.8, color="red")[0]
    ax0.axhline(y=0, color="black", linestyle="-")
    
    ax0.get_xaxis().set_ticklabels([])
    ax0.get_yaxis().set_ticklabels([])
    ax0.set_xlabel("statistics q(x), where x is data", fontsize=18,\
                   horizontalalignment='right', x=1.0)
    ax0.set_ylabel("probability density", fontsize=18)
    ax0.set_xlim(*rangeX)
    ax0.set_ylim(bottom=0.0)    

    dataLine = ax0.axvline(x=dataLoc, ymin=0.0, ymax=1.0, color="black", linewidth=2,\
                           linestyle="--")
    CRfill = ax0.fill_between(qVal[qVal_CRsel], baseline[qVal_CRsel],\
                              gausNull[qVal_CRsel], color='blue', alpha=0.5)
    PVFill = ax0.fill_between(qVal[qVal_PVsel], baseline[qVal_PVsel],\
                              gausNull[qVal_PVsel], color='blue', zorder=5)
    PSfill = ax0.fill_between(qVal[qVal_PSsel], baseline[qVal_PSsel],\
                              gausAlt[qVal_PSsel], color='red', alpha=0.5) 
    plotList = [[nullDist, altDist, dataLine, CRfill, PVFill, PSfill], \
                ["P(q|$\mu$=$\mu_s$) for signal null hypothesis",\
                 "P(q|$\mu$=$\mu_0$) for no-signal alternative hypothesis", 
                 "q(x$_{obs}$),\nx$_{obs}$: observed data",
                 "P(q>q$_\\alpha$|$\mu_s$) $\equiv \\alpha$,\n"+\
                 "q$_\\alpha$: critical value, $\\alpha$: significance set to 0.05",\
                 "P(q>q(x$_{obs}$)|$\mu_s$) $\equiv$ p$_\mu$,\np$_\mu$: p-value",\
                 "P(q>q(x$_{obs}$)|$\mu_0$) $\equiv$ M_$\mu_0$($\mu_s$),\n"+\
                 "M_$\mu_0$($\mu_s$): power/sensitivity"]]
    legObj = ax0.legend(*plotList, loc="upper right", fontsize=14)

    ylim = ax0.get_ylim()
    ax0.text(qAlpha,  ylim[0]-0.03*(ylim[1]-ylim[0]), "q$_\\alpha$", fontsize=18,\
             ha='center')
    ax0.text(dataLoc, ylim[1]-0.03*(ylim[1]-ylim[0]), "q(x$_{obs}$)", fontsize=18)
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__))
    filenameFig = exepath + "/hypoTestingDef.png"
    gs.tight_layout(fig)
    plt.savefig(filenameFig)
    if verbosity >= 1: print("Creating the following files:\n" + filenameFig)

#########################################################################################
TOLERANCE = pow(10.0, -10)
SNUMBER   = pow(10.0, -124)
def gaussian(mu, sig, x):
    X = np.array(x)
    vals = np.exp(-np.power(X-mu,2.0)/(2.0*np.power(sig,2.0)))\
         *(1.0/(sig*np.sqrt(2.0*np.pi)))
    vals[vals < SNUMBER] = SNUMBER
    return vals
#######################################################################################
if __name__ == "__main__":
    print("\n####################################################################Head")
    main()
    print("######################################################################Tail")




