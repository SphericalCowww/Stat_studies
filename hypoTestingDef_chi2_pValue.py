import os, sys, pathlib, time, re, glob, math, copy
from datetime import datetime
import warnings
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return "%s:%s: %s: %s\n" % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import optimize
from scipy import stats
from tqdm import tqdm
import iminuit
from iminuit import cost, Minuit

#####################################################################################################
def gaus_scipy(x, A, mu, sig):
    if (type(x) is list) or (type(x) is np.ndarray): X = np.array(x)
    else:                                            X = np.array([x])
    Y = A*np.array([stats.norm.pdf(Xval, loc=mu, scale=sig) for Xval in X])
    if len(Y) == 1: return Y[0]
    return Y
def eval_binnedPoisson_chi2(dataArr, fitArr):
    chi2Arr = np.power(dataArr-fitArr, 2)/fitArr
    chi2mask = (np.isnan(chi2Arr) == False)     # NOTE: nan values are removed for chi2 eval 
    return np.sum(chi2Arr[chi2mask])
def eval_binnedPoisson_chi2L(dataArr, fitArr):
    chi2Arr = 2*(fitArr - dataArr + dataArr*np.log(dataArr/fitArr))
    chi2mask = (np.isnan(chi2Arr) == False)     # NOTE: nan values are removed for chi2 eval 
    return np.sum(chi2Arr[chi2mask])
def poissonShuffle(countArr):
    shuffleArr = copy.deepcopy(countArr)
    for binIdx in range(len(shuffleArr)):
        shuffleArr[binIdx] = np.random.poisson(countArr[binIdx], size=1)[0]
    return shuffleArr 
def main():
    verbosity = 1
    
    np.random.seed(9)

    binN = 100
    rangeX = [-10.0, 10.0]

    dataMu  = 1.0
    dataSig = 2.0
    dataN   = 10000#100000000
    fitRange = [-4.0, 6.0]

    chi2_templateN = 10000
    rangeChi2 = [0, 6.0]
    pValN = 2001
    pValbinN = 100#max(1, int(pValN/10.0))

    xVals   = np.linspace(*rangeX, binN+1)[:-1]
    binSize = xVals[1] - xVals[0]
    fitRangeMask = ((fitRange[0] <= xVals)&(xVals < fitRange[1]))
    initPars = [(1.0*dataN)/binN, dataMu, dataSig]

    chi2Vals = np.linspace(*rangeChi2, 4*binN+1)[:-1]
    pValVals = np.linspace(0.0, 1.0, pValbinN+1)[:-1]

    chi2_template_poisson_binned_pVals = []
    for pValIter in tqdm(range(pValN)):
        dataSamp = np.random.normal(dataMu, dataSig, dataN)
        dataHist = np.histogram(dataSamp, bins=binN, range=rangeX)[0]   #bin2left  
        dataHistErr = [count if count > 0 else 1 for count in dataHist]

        costFunc = iminuit.cost.LeastSquares(xVals, dataHist, dataHistErr, gaus_scipy)
        costFunc.mask = fitRangeMask
        objMinuit = iminuit.Minuit(costFunc, *initPars)
        objMinuit.limits[1] = [-2.0, 2.0]
        objMinuit.limits[2] = [0.0, 10.0]
        optResult = objMinuit.migrad()

        fitPars = [val for val in optResult.values]
        fitErrs = [err for err in optResult.errors]
        fitCov  = [row for row in optResult.covariance]
        print("Multipeak fit:"); print(optResult)

        fitHist = gaus_scipy(xVals, *fitPars)
        ndof = binN + len(fitPars) - 1
        '''
        chi2_ndofs           = optResult.fmin.reduced_chi2 #optResult.fval/optResult.ndof # official
        chi2_multinominal    = iminuit.cost.multinominal_chi2(dataHist, fitHist)/ndof
        chi2_poisson         = iminuit.cost.poisson_chi2(     dataHist, fitHist)/ndof
        chi2_poisson_binned  = eval_binnedPoisson_chi2(       dataHist, fitHist)/ndof
        chi2_poisson_binnedL = eval_binnedPoisson_chi2L(dataHist, fitHist)/ndof
        '''
        chi2_poisson_binned = eval_binnedPoisson_chi2(dataHist, fitHist)/ndof
        if 1 == 1:#pValIter == 0:
            chi2_template_poisson_binned_temp, chi2_template_poisson_binned = [], []
            for templateIdx in tqdm(range(chi2_templateN)):
                shuffledHist = poissonShuffle(fitHist)
                chi2_template_poisson_binned_temp.append(shuffledHist)
                # NOTE: the following definition of template is vital to validate the p-value
                chi2_template_poisson_binned.append(eval_binnedPoisson_chi2(shuffledHist,fitHist)\
                                                    /ndof)
            chi2_template_poisson_binned_hist = np.histogram(chi2_template_poisson_binned,\
                                                              bins=4*binN, range=rangeChi2)[0]
        pValmask = (chi2_template_poisson_binned > chi2_poisson_binned)
        chi2_template_poisson_binned_pVals.append(np.sum(pValmask)/chi2_templateN)
        print("p-value =", chi2_template_poisson_binned_pVals[-1], '\n')
        pVal_hist = np.histogram(chi2_template_poisson_binned_pVals,\
                                 bins=pValbinN, range=[0.0, 1.0])[0]
        if (pValIter+1)%100 == 0:
            plotHist(pValIter+1, verbosity, binN, dataN, chi2_templateN, \
                     xVals, rangeX, chi2_template_poisson_binned_temp, dataHist, fitHist, fitPars,\
                     chi2Vals, rangeChi2, chi2_template_poisson_binned_hist, chi2_poisson_binned,\
                     pValVals, chi2_template_poisson_binned_pVals, pVal_hist)
#####################################################################################################
def plotHist(iterIdx, verbosity, binN, dataN, chi2_templateN,\
             xVals, rangeX, chi2_template_poisson_binned_temp, dataHist, fitHist, fitPars,\
             chi2Vals, rangeChi2, chi2_template_poisson_binned_hist, chi2_poisson_binned,\
             pValVals, chi2_template_poisson_binned_pVals, pVal_hist):
#plots
    gridSpec = [3, 1]
    figSize, marginRatio = getSizeMargin(gridSpec, subplotSize=[15.0, 7.0])
    fig = plt.figure(figsize=figSize); fig.subplots_adjust(*marginRatio)
    gs = gridspec.GridSpec(*gridSpec)
    matplotlib.rc("xtick", labelsize=24)
    matplotlib.rc("ytick", labelsize=24)
    ax = []
    for axIdx in range(gridSpec[0]*gridSpec[1]):
        ax.append(fig.add_subplot(gs[axIdx]));
        ax[-1].ticklabel_format(style="sci", scilimits=(-2, 2))
        ax[-1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[-1].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())   
    ###plot0
    axIdx = 0
    for template in chi2_template_poisson_binned_temp:
         ax[axIdx].plot(xVals, template, color="limegreen", drawstyle="steps-mid",\
                        linewidth=2, alpha=0.1)
    ax[axIdx].plot(xVals, dataHist, color="blue", drawstyle="steps-mid", linewidth=4)
    ax[axIdx].plot(xVals, fitHist, color="red", drawstyle="steps-mid", linewidth=3, alpha=0.7)
    ax[axIdx].plot(xVals, gaus_scipy(xVals, *fitPars), linewidth=2, alpha=1.0, color="red")
    textStr = "binN = " + str(binN) + "\neventN = "+str(dataN)
    ax[axIdx].text(0.03, 1.0, textStr, fontsize=24, transform=ax[axIdx].transAxes,\
                   color="blue", horizontalalignment="left", verticalalignment="top")
    ax[axIdx].set_xlabel("x", fontsize=24)
    ax[axIdx].set_ylabel("count", fontsize=24)
    ax[axIdx].set_xlim(*rangeX)
    ax[axIdx].set_ylim(bottom=0)

    ###plot1
    axIdx = 1
    plotObj = []
    plotObj.append(ax[axIdx].plot(chi2Vals, chi2_template_poisson_binned_hist, color="green",\
                                  drawstyle="steps-post", linewidth=4)[0])
    ax[axIdx].axvline(x=chi2_poisson_binned, color="red", linestyle="dashed", linewidth=3)
    ax[axIdx].text(0.6, 0.92, "templateN = "+str(chi2_templateN), fontsize=24,\
                   transform=ax[axIdx].transAxes, color="green", horizontalalignment="left",\
                   verticalalignment="bottom")
    ax[axIdx].set_xlabel("reduced chi2", fontsize=24)
    ax[axIdx].set_ylabel("count", fontsize=24)
    ax[axIdx].set_xlim(*rangeChi2)
    ax[axIdx].set_ylim(bottom=0)
    textStr  ="chi2/ndof = "+str(round(chi2_poisson_binned, 5)) + "\np-value = "
    textStr +=str(round(chi2_template_poisson_binned_pVals[-1],5))
    ax[axIdx].text(chi2_poisson_binned, ax[axIdx].get_ylim()[1], textStr,\
                   fontsize=18, color="red", horizontalalignment="left", verticalalignment="bottom")

    ###plot2
    axIdx = 2
    ax[axIdx].plot(pValVals, pVal_hist, color="red", drawstyle="steps-mid", linewidth=4)
    ax[axIdx].text(0.6, 1.0, "pValN = "+str(iterIdx), fontsize=24, transform=ax[axIdx].transAxes,\
                   color="red", horizontalalignment="left", verticalalignment="bottom")
    ax[axIdx].set_xlabel("p-value", fontsize=24)
    ax[axIdx].set_ylabel("count", fontsize=24)
    ax[axIdx].set_xlim(0.0, 1.0)
    ax[axIdx].set_ylim(bottom=0.9)
    ax[axIdx].set_yscale('log')
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__))
    pathlib.Path(exepath+"/figures").mkdir(parents=True, exist_ok=True)
    filenameFig = exepath + "/figures/hypoTestingDef_chi2_pValue"+str(iterIdx)+".png"
    gs.tight_layout(fig)
    plt.savefig(filenameFig)
    if verbosity >= 1: print("Creating the following files:\n" + filenameFig)
    plt.close('all')
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




