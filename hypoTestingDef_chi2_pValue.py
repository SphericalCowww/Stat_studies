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
    dataN   = 100000000
    fitRange = [-4.0, 6.0]

    chi2_templateN = 10000
    rangeChi2 = [0, 6.0]

    dataSamp = np.random.normal(dataMu, dataSig, dataN)
    xVals   = np.linspace(*rangeX, binN+1)[:-1]
    binSize = xVals[1] - xVals[0]
    dataHist = np.histogram(dataSamp, bins=binN, range=rangeX)[0]   #bin2left  

    fitRangeMask = ((fitRange[0] <= xVals)&(xVals < fitRange[1]))
    dataHistErr = [count if count > 0 else 1 for count in dataHist]

    initPars = [(1.0*dataN)/binN, dataMu, dataSig]
    costFunc = iminuit.cost.LeastSquares(xVals, dataHist, dataHistErr, gaus_scipy)
    costFunc.mask = fitRangeMask
    objMinuit = iminuit.Minuit(costFunc, *initPars)
    objMinuit.limits[1] = [-2.0, 2.0]
    objMinuit.limits[2] = [0.0, 10.0]
    optResult = objMinuit.migrad()

    fitPars = [val for val in optResult.values]
    fitErrs = [err for err in optResult.errors]
    fitCov  = [row for row in optResult.covariance]
    print('Multipeak fit:'); print(optResult)

    chi2Vals = np.linspace(*rangeChi2, binN+1)[:-1]

    fitHist = gaus_scipy(xVals, *fitPars)
    ndof = binN + len(fitPars) - 1
    chi2_ndofs           = optResult.fmin.reduced_chi2 #optResult.fval/optResult.ndof    # official
    chi2_multinominal    = iminuit.cost.multinominal_chi2(dataHist, fitHist)/ndof
    chi2_poisson         = iminuit.cost.poisson_chi2(     dataHist, fitHist)/ndof
    chi2_poisson_binned  = eval_binnedPoisson_chi2(       dataHist, fitHist)/ndof
    chi2_poisson_binnedL = eval_binnedPoisson_chi2L(      dataHist, fitHist)/ndof

    chi2_template_poisson_binnedL_temp, chi2_template_poisson_binnedL = [], []
    chi2_template_poisson_binnedL = []
    
    for templateIdx in tqdm(range(chi2_templateN)):
        shuffledHist = poissonShuffle(fitHist)
        chi2_template_poisson_binnedL_temp.append(shuffledHist)
        # NOTE: the following definition of template is vital to validate the p-value
        chi2_template_poisson_binnedL.append(eval_binnedPoisson_chi2(shuffledHist, fitHist)/ndof)
    chi2_template_poisson_binnedL_hist = np.histogram(chi2_template_poisson_binnedL,\
                                                      bins=binN, range=rangeChi2)[0] 
#####################################################################################################
#plots
    gridSpec = [2, 1]
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
    for template in chi2_template_poisson_binnedL_temp:
         ax[axIdx].plot(xVals, template, color="limegreen", drawstyle="steps-mid",\
                        linewidth=2, alpha=0.1)
    ax[axIdx].plot(xVals, dataHist, color="blue", drawstyle="steps-mid", linewidth=4)
    ax[axIdx].plot(xVals, fitHist, color="red", drawstyle="steps-mid", linewidth=3, alpha=0.7)
    ax[axIdx].plot(xVals, gaus_scipy(xVals, *fitPars), linewidth=2, alpha=1.0, color="red")
    textStr = "binN = " + str(binN) + "\neventN = "+str(dataN)
    ax[axIdx].text(0.03, 1.0, textStr, fontsize=24, transform=ax[axIdx].transAxes,\
                   color='blue', horizontalalignment='left', verticalalignment='top')
    ax[axIdx].set_xlabel('x', fontsize=24)
    ax[axIdx].set_ylabel('count', fontsize=24)
    ax[axIdx].set_ylim(bottom=0)

    ###plot1
    axIdx = 1
    plotObj = []
    plotObj.append(ax[axIdx].plot(chi2Vals, chi2_template_poisson_binnedL_hist, color="green",\
                                  drawstyle="steps-post", linewidth=4)[0])
    ax[axIdx].axvline(x=chi2_poisson_binnedL, color='red', linestyle='dashed', linewidth=3)
    ax[axIdx].text(0.6, 0.92, "templateN = "+str(chi2_templateN), fontsize=24,\
                   transform=ax[axIdx].transAxes, color='green', horizontalalignment='left',\
                   verticalalignment='bottom')
    ax[axIdx].set_xlabel('reduced chi2', fontsize=24)
    ax[axIdx].set_ylabel('count', fontsize=24)
    ax[axIdx].set_ylim(bottom=0)
    textStr  ="chi2/ndof = "+str(round(chi2_poisson_binnedL, 5)) + "\np-value = "
    textStr +=str(round(np.sum(chi2_template_poisson_binnedL>chi2_poisson_binnedL)/chi2_templateN,5))
    ax[axIdx].text(chi2_poisson_binnedL, ax[axIdx].get_ylim()[1], textStr,\
                   fontsize=18, color='red', horizontalalignment='left', verticalalignment='bottom')
#save plots
    exepath = os.path.dirname(os.path.abspath(__file__))
    filenameFig = exepath + "/hypoTestingDef_chi2_pValue.png"
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




