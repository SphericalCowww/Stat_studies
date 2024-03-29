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

#####################################################################################################
def main():
    verbosity = 1
    rangeX = [-7, 22]
    plotRes = 1000

    
    muSigNull = [0.0, 1.8]
    muSigAlt  = [5.9, 2.5]
    dataLoc = 2.4
    rangeX = [-7, 22]
    

    '''
    muSigNull = [0.0, 3.5]
    muSigAlt  = [8.0, 0.5]
    dataLoc = 2.4
    rangeX = [-7, 22]
    '''

    '''
    trueSignalN = 2
    backgroundN = 10
    signalN     = 5
    muSigNull = [-(backgroundN+signalN), np.sqrt(backgroundN+signalN)]
    muSigAlt  = [-backgroundN, np.sqrt(backgroundN)]
    dataLoc = -(backgroundN)
    rangeX = [-2.5*backgroundN, 2*backgroundN]
    '''

#####################################################################################################
    qVal     = np.linspace(*rangeX, plotRes)
    baseline = qVal*0.0
    gausNull = gaussian(*muSigNull, qVal)
    gausAlt  = gaussian(*muSigAlt,  qVal)

    qAlpha = muSigNull[0]+2*muSigNull[1]    #2 sigma-significance
    qVal_CRsel  = (qVal > qAlpha)           #CR for critical region
    qVal_PVsel  = (qVal > dataLoc)          #PV for p-value
    qVal_PSsel  = (qVal > qAlpha)           #PS for power/sensitivity
    qVal_CLssel = (qVal < dataLoc)          #CLs weight
#plots
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    
    nullDist = ax0.plot(qVal, gausNull, linewidth=2, alpha=0.8, color="blue")[0]
    altDist  = ax0.plot(qVal, gausAlt,  linewidth=2, alpha=0.8, color="red")[0]
    ax0.axhline(y=0, color="black", linestyle="-")
    
    ax0.get_xaxis().set_ticklabels([])
    ax0.get_yaxis().set_ticklabels([])
    ax0.set_xlabel("statistics q($\mu$), where $\mu$ is the fit parameters to data",\
                   fontsize=18, horizontalalignment='right', x=1.0)
    ax0.set_ylabel("probability density", fontsize=18)
    ax0.set_xlim(*rangeX)
    ax0.set_ylim(bottom=0.0)    

    dataLine = ax0.axvline(x=dataLoc, ymin=0.0, ymax=1.0, color="black", linewidth=2, linestyle="--")
    CRfill  = ax0.fill_between(qVal[qVal_CRsel], baseline[qVal_CRsel], gausNull[qVal_CRsel],\
                               color="blue", alpha=0.5, zorder=0)
    PVfill  = ax0.fill_between(qVal[qVal_PVsel], baseline[qVal_PVsel], gausNull[qVal_PVsel],\
                               color="none", hatch="///", edgecolor='blue')
    PSfill  = ax0.fill_between(qVal[qVal_PSsel], baseline[qVal_PSsel], gausAlt[qVal_PSsel],\
                               color="none", alpha=0.5, hatch="\\\\\\", edgecolor='red') 
    CLsfill = ax0.fill_between(qVal[qVal_CLssel], baseline[qVal_CLssel], gausAlt[qVal_CLssel],\
                               color="none", alpha=0.5, hatch="----", edgecolor='red')
    blank = ax0.axvline(x=dataLoc, color="black", alpha=0.0)
    plotList = [[blank, nullDist, altDist, dataLine, CRfill, PVfill, PSfill, CLsfill, blank, blank],\
                ["With q defined via signal_null/no-signal_alternative hypothesis:",\
                 "P$_{null}$(q|$\mu$=$\mu_s$)",\
                 "P$_{alt}$(q|$\mu$=$\mu_0$)", 
                 "q($\hat{\mu}_{data}$),\n"+\
                 "$\hat{\mu}_{data}$: parameters $\mu$ as fit to observed data",
                 "P$_{null}$(q>q$_\\alpha$|$\mu_s$) $\equiv \\alpha$, set to say 0.05,\n"+\
                 "q$_\\alpha$: critical value, "+\
                 "$\\alpha$: significance/size/false positive/type-I error",\
                 "P$_{null}$(q>q($\hat{\mu}_{data}$)|$\mu_s$) $\equiv$ p$_\mu$,\np$_\mu$: p-value",\
                 "P$_{alt}$(q>q$_\\alpha$|$\mu_0$) $\equiv$ M$_{\mu_0}$($\mu_s$)"+\
                 "$\equiv$ 1-$\\beta$,\nM$_{\mu_0}$($\mu_s$): power/sensitivity, "+\
                 "$\\beta$: false negative/type-II error",\
                 "P$_{alt}$(q<q($\hat{\mu}_{data}$)|$\mu_0$) $\equiv$ p$_0$,\n"+\
                 "1/(1-p$_0$): CL$_s$ penalty factor",\
                 "\nP($[\mu_0, \mu_1]\\ni\mu|\mu=\mu_{true}$): coverage probability",\
                 "\nWith q defined via no-signal_null/signal_alternative hypothesis: \n"+\
                 "$\mu_d$: discovery potential, if "+\
                 "P$_{null}$(q>q($\mu_d$)|$\mu$=$\mu_0$) $\equiv \\alpha$ = 0.05"]]
    legObj = ax0.legend(*plotList, loc="upper right", fontsize=12)

    ylim = ax0.get_ylim()
    ax0.text(qAlpha,  ylim[0]-0.03*(ylim[1]-ylim[0]), "q$_\\alpha$", fontsize=18,\
             ha='center')
    ax0.text(dataLoc, ylim[1]-0.04*(ylim[1]-ylim[0]), "q($\hat{\mu}_{data}$)", fontsize=18)
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__))
    filenameFig = exepath + "/hypoTestingDef.png"
    gs.tight_layout(fig)
    plt.savefig(filenameFig)
    if verbosity >= 1: print("Creating the following files:\n" + filenameFig)

#####################################################################################################
TOLERANCE = pow(10.0, -10)
SNUMBER   = pow(10.0, -124)
def gaussian(mu, sig, x):
    X = np.array(x)
    vals = np.exp(-np.power(X-mu,2.0)/(2.0*np.power(sig,2.0)))\
         *(1.0/(sig*np.sqrt(2.0*np.pi)))
    vals[vals < SNUMBER] = SNUMBER
    return vals
#####################################################################################################
if __name__ == "__main__":
    print("\n####################################################################Head")
    main()
    print("######################################################################Tail")




