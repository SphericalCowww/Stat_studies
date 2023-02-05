import os, sys, pathlib, time, re, glob, math, copy
from datetime import datetime
import warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
import matplotlib
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

    detectionN = 12
    backgroundN = 10

    detectionN  = -detectionN
    backgroundN = -backgroundN
    rangeX = [4.0*backgroundN, -0.5*backgroundN]
    
    muNoSigAlt_1 = [backgroundN, np.sqrt(np.abs(backgroundN))]
    sigN = 1.0
    meanNull_11  = -np.power((-sigN-np.sqrt(np.power(sigN, 2)-4*detectionN))/2, 2)
    muSigNull_11 = [meanNull_11, np.sqrt(np.abs(meanNull_11))]
    qAlpha_11    = muSigNull_11[0] + sigN*muSigNull_11[1]
    sigN = 2.0
    meanNull_12  = -np.power((-sigN-np.sqrt(np.power(sigN, 2)-4*detectionN))/2, 2)
    muSigNull_12 = [meanNull_12, np.sqrt(np.abs(meanNull_12))]
    qAlpha_12    = muSigNull_12[0] + sigN*muSigNull_12[1]
    sigN = 3.0
    meanNull_13  = -np.power((-sigN-np.sqrt(np.power(sigN, 2)-4*detectionN))/2, 2)
    muSigNull_13 = [meanNull_13, np.sqrt(np.abs(meanNull_13))]
    qAlpha_13    = muSigNull_13[0] + sigN*muSigNull_13[1]

    muNoSigNull_2 = [backgroundN, np.sqrt(np.abs(backgroundN))]
    qAlpha_21 = muNoSigNull_2[0] - 1.0*muNoSigNull_2[1]
    qAlpha_22 = muNoSigNull_2[0] - 2.0*muNoSigNull_2[1]
    qAlpha_23 = muNoSigNull_2[0] - 3.0*muNoSigNull_2[1]
 
#####################################################################################################
#plots
    gridSpec = [4, 1]
    figSize, marginRatio = getSizeMargin(gridSpec, subplotSize=[15.0, 7.0])
    fig = plt.figure(figsize=figSize); fig.subplots_adjust(*marginRatio)
    gs = gridspec.GridSpec(*gridSpec)
    matplotlib.rc('xtick', labelsize=24)
    matplotlib.rc('ytick', labelsize=24)
    ax = []
    for axIdx in range(gridSpec[0]*gridSpec[1]):
        ax.append(fig.add_subplot(gs[axIdx]));
        ax[-1].ticklabel_format(style='sci', scilimits=(-2, 2))
        ax[-1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[-1].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())   
    ###plot0
    axIdx = 0
    qVal     = np.linspace(*rangeX, plotRes)
    baseline = qVal*0.0
    gausAlt  = gaussian(*muNoSigAlt_1, qVal)
    gausNull = gaussian(*muSigNull_11, qVal)
    qVal_CRsel = (qVal > qAlpha_11)         #CR for critical region
    qVal_PVsel = (qVal > detectionN)        #PV for p-value
    qVal_PSsel = (qVal > qAlpha_11)         #PS for power/sensitivity
  
    altDist  = ax[axIdx].plot(qVal, gausAlt,  linewidth=2, alpha=0.8, color="red")[0]
    nullDist = ax[axIdx].plot(qVal, gausNull, linewidth=2, alpha=0.8, color="blue")[0]
    ax[axIdx].axhline(y=0, color="black", linestyle="-")
    
    ax[axIdx].get_xaxis().set_ticklabels([])
    ax[axIdx].get_yaxis().set_ticklabels([])
    ax[axIdx].set_xlabel("statistics q($\mu$)", fontsize=18, horizontalalignment='right', x=1.0)
    ax[axIdx].set_ylabel("probability density", fontsize=18)
    ax[axIdx].set_xlim(*rangeX)
    ax[axIdx].set_ylim(bottom=0.0)

    detLine        = ax[axIdx].axvline(x=detectionN,  ymin=0.0, ymax=1.0, color="black",linewidth=3,\
                                       linestyle="--")
    meanLineSideL  = ax[axIdx].axvline(x=meanNull_11, ymin=0.0, ymax=1.0, color="blue", linewidth=3,\
                                       linestyle="--")
    meanLineCenter = ax[axIdx].axvline(x=meanNull_12, ymin=0.0, ymax=1.0, color="green",linewidth=4,\
                                       linestyle="-")
    meanLineSideR  = ax[axIdx].axvline(x=meanNull_13, ymin=0.0, ymax=1.0, color="green",linewidth=3,\
                                       linestyle="--")
    meanLineFill = ax[axIdx].axvspan(meanNull_11, meanNull_13, color="green", alpha=0.2, zorder=0)
    CRfill = ax[axIdx].fill_between(qVal[qVal_CRsel], baseline[qVal_CRsel], gausNull[qVal_CRsel],\
                                    color="blue", alpha=0.5, zorder=0)
    PVfill = ax[axIdx].fill_between(qVal[qVal_PVsel], baseline[qVal_PVsel], gausNull[qVal_PVsel],\
                                    color="none", hatch="///", edgecolor='blue')
    PSfill = ax[axIdx].fill_between(qVal[qVal_PSsel], baseline[qVal_PSsel], gausAlt[qVal_PSsel],\
                                    color="none", alpha=0.5, hatch="\\\\\\", edgecolor='red') 
    blank  = ax[axIdx].axvline(x=detectionN, ymin=0.0, ymax=1.0, color="black", alpha=0.0)
    
    ylim = ax[axIdx].get_ylim()
    ax[axIdx].text(qAlpha_11,  ylim[0]-0.03*(ylim[1]-ylim[0]), "q$_\\alpha$",fontsize=18,ha="center")
    ax[axIdx].text(detectionN, ylim[1]-0.04*(ylim[1]-ylim[0]), "q($\hat{\mu}_{data}$)", fontsize=18)

    plotList = [[nullDist, altDist, detLine, meanLineSideL, CRfill, PVfill, PSfill, meanLineFill], \
                ["P$_{null}$(q|$\mu$=$\mu_s$)",\
                 "P$_{alt}$(q|$\mu$=$\mu_0$)",\
                 "q($\hat{\mu}_{data}$) = q$_\\alpha$",\
                 "mean(P$_{null}$)",\
                 "$\\alpha$="+str(round(1.0-stats.norm.cdf(1.0), 5))+" (1$\\sigma$)",\
                 "p-value",\
                 "power",\
                 "Brazilian band\n(sensitivity)"]]
    legObj = ax[axIdx].legend(*plotList, loc="upper right", fontsize=12)

    ###plot1
    axIdx = 1
    qVal     = np.linspace(*rangeX, plotRes)
    baseline = qVal*0.0
    gausAlt  = gaussian(*muNoSigAlt_1, qVal)
    gausNull = gaussian(*muSigNull_12, qVal)
    qVal_CRsel = (qVal > qAlpha_12)         #CR for critical region
    qVal_PVsel = (qVal > detectionN)        #PV for p-value
    qVal_PSsel = (qVal > qAlpha_12)         #PS for power/sensitivity
 
    altDist  = ax[axIdx].plot(qVal, gausAlt,  linewidth=2, alpha=0.8, color="red")[0]
    nullDist = ax[axIdx].plot(qVal, gausNull, linewidth=2, alpha=0.8, color="blue")[0]
    ax[axIdx].axhline(y=0, color="black", linestyle="-")
    
    ax[axIdx].get_xaxis().set_ticklabels([])
    ax[axIdx].get_yaxis().set_ticklabels([])
    ax[axIdx].set_xlabel("statistics q($\mu$)", fontsize=18, horizontalalignment='right', x=1.0)
    ax[axIdx].set_ylabel("probability density", fontsize=18)
    ax[axIdx].set_xlim(*rangeX)
    ax[axIdx].set_ylim(bottom=0.0)

    detLine = ax[axIdx].axvline(x=detectionN, ymin=0.0, ymax=1.0, color="black", linewidth=3,\
                                linestyle="--")
    meanLineSideL  = ax[axIdx].axvline(x=meanNull_11, ymin=0.0, ymax=1.0, color="green",linewidth=3,\
                                       linestyle="--")
    meanLineCenter = ax[axIdx].axvline(x=meanNull_12, ymin=0.0, ymax=1.0, color="blue", linewidth=4,\
                                       linestyle="-")
    meanLineSideR  = ax[axIdx].axvline(x=meanNull_13, ymin=0.0, ymax=1.0, color="green",linewidth=3,\
                                   linestyle="--")
    meanLineFill   = ax[axIdx].axvspan(meanNull_11, meanNull_13, color="green", alpha=0.2, zorder=0)
    CRfill = ax[axIdx].fill_between(qVal[qVal_CRsel], baseline[qVal_CRsel], gausNull[qVal_CRsel],\
                                    color="blue", alpha=0.5, zorder=0)
    PVfill = ax[axIdx].fill_between(qVal[qVal_PVsel], baseline[qVal_PVsel], gausNull[qVal_PVsel],\
                                    color="none", hatch="///", edgecolor='blue')
    PSfill = ax[axIdx].fill_between(qVal[qVal_PSsel], baseline[qVal_PSsel], gausAlt[qVal_PSsel],\
                                    color="none", alpha=0.5, hatch="\\\\\\", edgecolor='red')
    blank  = ax[axIdx].axvline(x=detectionN, ymin=0.0, ymax=1.0, color="black", alpha=0.0)

    ylim = ax[axIdx].get_ylim()
    ax[axIdx].text(qAlpha_12,  ylim[0]-0.03*(ylim[1]-ylim[0]), "q$_\\alpha$",fontsize=18,ha="center")
    ax[axIdx].text(detectionN, ylim[1]-0.04*(ylim[1]-ylim[0]), "q($\hat{\mu}_{data}$)", fontsize=18)

    plotList = [[nullDist, altDist, detLine, meanLineCenter, CRfill], \
                ["P$_{null}$(q|$\mu$=$\mu_s$)",\
                 "P$_{alt}$(q|$\mu$=$\mu_0$)",\
                 "q($\hat{\mu}_{data}$) = q$_\\alpha$",\
                 "mean(P$_{null}$)",\
                 "$\\alpha$="+str(round(1.0-stats.norm.cdf(2.0), 5))+" (2$\\sigma$)"]]
    legObj = ax[axIdx].legend(*plotList, loc="upper right", fontsize=12)

    ###plot2
    axIdx = 2
    qVal     = np.linspace(*rangeX, plotRes)
    baseline = qVal*0.0
    gausAlt  = gaussian(*muNoSigAlt_1, qVal)
    gausNull = gaussian(*muSigNull_13, qVal)
    qVal_CRsel = (qVal > qAlpha_13)         #CR for critical region
    qVal_PVsel = (qVal > detectionN)        #PV for p-value
    qVal_PSsel = (qVal > qAlpha_13)         #PS for power/sensitivity

    altDist  = ax[axIdx].plot(qVal, gausAlt,  linewidth=2, alpha=0.8, color="red")[0]
    nullDist = ax[axIdx].plot(qVal, gausNull, linewidth=2, alpha=0.8, color="blue")[0]
    ax[axIdx].axhline(y=0, color="black", linestyle="-")
    
    ax[axIdx].get_xaxis().set_ticklabels([])
    ax[axIdx].get_yaxis().set_ticklabels([])
    ax[axIdx].set_xlabel("statistics q($\mu$)", fontsize=18, horizontalalignment='right', x=1.0)
    ax[axIdx].set_ylabel("probability density", fontsize=18)
    ax[axIdx].set_xlim(*rangeX)
    ax[axIdx].set_ylim(bottom=0.0)

    detLine = ax[axIdx].axvline(x=detectionN, ymin=0.0, ymax=1.0, color="black", linewidth=3,\
                                linestyle="--")
    meanLineSideL  = ax[axIdx].axvline(x=meanNull_11, ymin=0.0, ymax=1.0, color="green",linewidth=3,\
                                       linestyle="--")
    meanLineCenter = ax[axIdx].axvline(x=meanNull_12, ymin=0.0, ymax=1.0, color="green",linewidth=4,\
                                       linestyle="-")
    meanLineSideR  = ax[axIdx].axvline(x=meanNull_13, ymin=0.0, ymax=1.0, color="blue", linewidth=3,\
                                       linestyle="--")
    meanLineFill   = ax[axIdx].axvspan(meanNull_11, meanNull_13, color="green", alpha=0.2, zorder=0)
    CRfill = ax[axIdx].fill_between(qVal[qVal_CRsel], baseline[qVal_CRsel], gausNull[qVal_CRsel],\
                                    color="blue", alpha=0.5, zorder=0)
    PVfill = ax[axIdx].fill_between(qVal[qVal_PVsel], baseline[qVal_PVsel], gausNull[qVal_PVsel],\
                                    color="none", hatch="///", edgecolor='blue')
    PSfill = ax[axIdx].fill_between(qVal[qVal_PSsel], baseline[qVal_PSsel], gausAlt[qVal_PSsel],\
                                    color="none", alpha=0.5, hatch="\\\\\\", edgecolor='red')
    blank  = ax[axIdx].axvline(x=detectionN, ymin=0.0, ymax=1.0, color="black", alpha=0.0)

    ylim = ax[axIdx].get_ylim()
    ax[axIdx].text(qAlpha_13,  ylim[0]-0.03*(ylim[1]-ylim[0]), "q$_\\alpha$",fontsize=18,ha="center")
    ax[axIdx].text(detectionN, ylim[1]-0.04*(ylim[1]-ylim[0]), "q($\hat{\mu}_{data}$)", fontsize=18)

    plotList = [[nullDist, altDist, detLine, meanLineSideR, CRfill],\
                ["P$_{null}$(q|$\mu$=$\mu_s$)",\
                 "P$_{alt}$(q|$\mu$=$\mu_0$)",\
                 "q($\hat{\mu}_{data}$) = q$_\\alpha$",\
                 "mean(P$_{null}$)",\
                 "$\\alpha$="+str(round(1.0-stats.norm.cdf(3.0), 5))+" (3$\\sigma$)"]]
    legObj = ax[axIdx].legend(*plotList, loc="upper right", fontsize=12)

    ###plot3
    axIdx = 3
    qVal     = np.linspace(*rangeX, plotRes)
    baseline = qVal*0.0
    gausNull = gaussian(*muNoSigNull_2, qVal)
    qVal_CRsel = (qVal < qAlpha_22)         #CR for critical region
    qVal_PVsel = (qVal < detectionN)        #PV for p-value
  
    nullDist = ax[axIdx].plot(qVal, gausNull, linewidth=2, alpha=0.8, color="blue")[0]
    ax[axIdx].axhline(y=0, color="black", linestyle="-")
    
    ax[axIdx].get_xaxis().set_ticklabels([])
    ax[axIdx].get_yaxis().set_ticklabels([])
    ax[axIdx].set_xlabel("statistics q($\mu$)", fontsize=18, horizontalalignment="right", x=1.0)
    ax[axIdx].set_ylabel("probability density", fontsize=18)
    ax[axIdx].set_xlim(*rangeX)
    ax[axIdx].set_ylim(bottom=0.0)

    detLine        = ax[axIdx].axvline(x=detectionN,  ymin=0.0, ymax=1.0, color="black",linewidth=3,\
                                       linestyle="--")
    alphaLineSideL  = ax[axIdx].axvline(x=qAlpha_21, ymin=0.0, ymax=1.0,color="green",linewidth=3,\
                                        linestyle="--")
    alphaLineCenter = ax[axIdx].axvline(x=qAlpha_22, ymin=0.0, ymax=1.0,color="green",linewidth=4,\
                                        linestyle="-")
    alphaLineSideR  = ax[axIdx].axvline(x=qAlpha_23, ymin=0.0, ymax=1.0,color="green",linewidth=3,\
                                        linestyle="--")
    alphaLineFill = ax[axIdx].axvspan(qAlpha_21, qAlpha_23, color="green", alpha=0.2, zorder=0)
    CRfill = ax[axIdx].fill_between(qVal[qVal_CRsel], baseline[qVal_CRsel], gausNull[qVal_CRsel],\
                                color="blue", alpha=0.5, zorder=0)
    PVfill = ax[axIdx].fill_between(qVal[qVal_PVsel], baseline[qVal_PVsel], gausNull[qVal_PVsel],\
                                color="none", hatch="///", edgecolor='blue')
    blank  = ax[axIdx].axvline(x=detectionN, ymin=0.0, ymax=1.0, color="black", alpha=0.0)
    
    ylim = ax[axIdx].get_ylim()
    ax[axIdx].text(qAlpha_22,  ylim[0]-0.03*(ylim[1]-ylim[0]),"q$_\\alpha$",fontsize=18,ha="center")
    ax[axIdx].text(detectionN, ylim[1]-0.04*(ylim[1]-ylim[0]), "q($\hat{\mu}_{data}$)", fontsize=18)

    plotList = [[nullDist, detLine, alphaLineSideL, CRfill, PVfill, alphaLineFill], \
                ["P$_{null}$(q|$\mu$=$\mu_0$)",\
                 "q($\hat{\mu}_{data}$) = q$_\\alpha$",\
                 "mean(P$_{null}$)",\
                 "$\\alpha$="+str(round(1.0-stats.norm.cdf(2.0), 5))+" (2$\\sigma$)",\
                 "p-value",\
                 "Brazilian band\n(discovery potential)"]]
    legObj = ax[axIdx].legend(*plotList, loc="upper right", fontsize=12)
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__))
    filenameFig = exepath + "/hypoTestingDef_BrazilianBand.png"
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
#plotMargin: left, bottom, right, top
DEFAULT_SCALEYX=((1.0-0.1-0.11)*7.0)/((1.0-0.13-0.08)*9.0)
def getSizeMargin(gridSpec, subplotSize=[9.0, 7.0], plotMargin=[0.13, 0.1, 0.08, 0.11]):
    figSize = (gridSpec[1]*subplotSize[0], gridSpec[0]*subplotSize[1])
    marginRatio = [plotMargin[0]/gridSpec[1], plotMargin[1]/gridSpec[0],\
                   1.0 - plotMargin[2]/gridSpec[1], 1.0 - plotMargin[3]/gridSpec[0],\
                   (plotMargin[0] + plotMargin[2])/(1.0 - plotMargin[0] - plotMargin[2]),\
                   (plotMargin[1] + plotMargin[3])/(1.0 - plotMargin[1] - plotMargin[3])]
    return figSize, marginRatio
#####################################################################################################
if __name__ == "__main__":
    print("\n####################################################################Head")
    main()
    print("######################################################################Tail")




