import os, sys, pathlib, time, re, glob, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

#pythonhosted.org/uncertainties/user_guide.html
#pydoc uncertainties.umath
from uncertainties import ufloat
import uncertainties.umath as umath
import uncertainties

#math.stackexchange.com/questions/547671
#######################################################################################
def main():
    ######Q1: how to get covariance given only the variable mean and std?
    # => (assumed answer unless told otherwise) Not possible unless assume Gaussian
    #standard error
    x = ufloat(2.0, 1.0)
    y = ufloat(3.0, 1.0)
    z = 3*umath.pow(x, 2) + 4*y
    #z = 3*umath.sqrt(x) + 4*y
    print(z)
    #does the following assume Gaussian uncertainty?
    #otherwise, need E(x^3) for z=3x^2+4y and
    #math.stackexchange.com/questions/547671
    print(uncertainties.correlation_matrix([x, y, z]))
    #also interesting, see Uncorrelatedness and independence:
    #en.wikipedia.org/wiki/Covariance
    print("--------------------------------------------------------------------------")
    
    ######Q2: assuming y = Kx, with K fixed
    #measuring x and y indepedently, and want z = x + y
    #does Var(z) = Var(x) + Var(y)?
    # =>
    x = ufloat(1.05, 0.05)
    y = ufloat(2.03, 0.05)
    K = y.n/x.n
    z = x + y
    z_x = (1.0 + K)*x
    z_y = (1.0/K + 1.0)*y
    print(np.sqrt(pow(x.s, 2) + pow(y.s, 2)))
    print(z.format("10.5f"))
    print(z_x.format("10.5f"))
    print(z_y.format("10.5f"))
    print(weightedAverage([z_x, z_y]).format("10.5f"))
    print(((z_x+z_y)/2.0).format("10.5f"))
    print("--------------------------------------------------------------------------")

    ######Q3: if x = A and y = 2A (e.g. voltage at 2 different site of a circuit), then
    #are x and y correlated?
    # => (assumed answer unless told otherwise) No, x and y are measurements, only the
    #paremeters of a fit should have correlation amoung other fit parameters.
    #A simultaneous fit between x and y/2 should just yeild A as the "uncorrelated" 
    #weighted average of the 2 measurements  

    #NOTE: if we fit to a constant function y = A, then the point estimate and the 
    #standard error of A follows that of a weighted average.
    #Refer to P. Bevington and D. Robinson, Eq.6.1 and Eq.6.21 assuming slope is 0\pm0

    #NOTE: correlations from the fit should be carried when estimating the error of
    #the fit function
    import iminuit
    from iminuit import cost, Minuit
    x    = [1.1, 1.9, 4.0]
    y    = [3.4, 5.5, 9.4] 
    yerr = [1.2, 0.9, 1.3]
    costFunc = iminuit.cost.LeastSquares(x, y, yerr, func_lin)
    objMinuit = iminuit.Minuit(costFunc, 1.0, 2.0)
    optResult = objMinuit.migrad()
    optPars = [val for val in optResult.values]
    optCov  = [row for row in optResult.covariance]
    #crucial step: stackoverflow.com/questions/64858229
    optPars = list(uncertainties.correlated_values(optPars, optCov)) 

    #linear fit err
    rangeRatio = 0.2
    funcX = np.linspace(x[0]-rangeRatio*(x[-1]-x[0]),\
                        x[-1]+rangeRatio*(x[-1]-x[0]), 1000)
    fitFunc = np.array([item.n for item in func_lin(funcX, *optPars)])
    fitFuncLower = fitFunc - np.array([item.s for item in func_lin(funcX, *optPars)])
    fitFuncUpper = fitFunc + np.array([item.s for item in func_lin(funcX, *optPars)])
    #plot
    gridSpec = [1, 1]
    figSize, marginRatio = getSizeMargin(gridSpec, subplotSize=[18.0, 10.0],\
                                         plotMargin=[0.08, 0.1, 0.04, 0.1])
    fig = plt.figure(figsize=figSize); fig.subplots_adjust(*marginRatio)
    gs = gridspec.GridSpec(*gridSpec)
    matplotlib.rc('xtick', labelsize=24)
    matplotlib.rc('ytick', labelsize=24)
    ax = []
    for axIdx in range(gridSpec[0]*gridSpec[1]):
        ax.append(fig.add_subplot(gs[axIdx]));
        ax[-1].ticklabel_format(style='sci', scilimits=(-3, 3))
        ax[-1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[-1].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[-1].xaxis.set_tick_params(length=6, width=1, which='major')
        ax[-1].xaxis.set_tick_params(length=3, width=1, which='minor')
        ax[-1].yaxis.set_tick_params(length=6, width=1, which='major')
        ax[-1].yaxis.set_tick_params(length=3, width=1, which='minor')

    inputPars ={'markersize':10.0,'markeredgewidth':3.0,'capsize':5.0,'elinewidth':4.0}
    ax[0].errorbar(x, y, yerr=yerr, color='blue', fmt='o', zorder=2, **inputPars) 
    ax[0].set_title('Q3 linear fit', fontsize=36, y=1.03)
    ax[0].set_xlabel('x', fontsize=32)
    ax[0].set_ylabel('y', fontsize=32)
    ax[0].plot(funcX, fitFunc,      linestyle='-',  linewidth=2,color='green',zorder=1)
    ax[0].plot(funcX, fitFuncLower, linestyle='--', linewidth=2,color='green',zorder=1)
    ax[0].plot(funcX, fitFuncUpper, linestyle='--', linewidth=2,color='green',zorder=1)
    ax[0].fill_between(funcX, fitFuncLower, fitFuncUpper, color='palegreen',\
                       alpha=0.5, zorder=0)
    plt.savefig('Q3.png')
    plt.clf()
    print('saving: Q3.png')
    print("--------------------------------------------------------------------------")

    #######Q4: standard error of weighted average with correlation?
    # => 
    #NOTE: python package 'uncertainties' doesn't have weighted average implemented:
    #stackoverflow.com/questions/43637370
    print("--------------------------------------------------------------------------")

    #######Q5: for rate measurement of total count N and total time T, why does the
    #weighted sum of smaller and smaller seperation yield smaller error bar?
    # => No, with correct calculation (easiest to assume equipartition), they should
    #be equivalent
    n = [12,  9,   12,  11,  8,   8,   12,  14,  10,  10,  19,  32]
    t = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0]
    n1 = [ufloat(count, np.sqrt(count)) for count in n]
    r1 = weightedAverage(np.array(n1)/np.array(t))
    print(r1, np.array(n1)/np.array(t))

    n2 = [n[0]+n[1], n[2]+n[3], n[4]+n[5], n[6]+n[7], n[8]+n[9], n[10]+n[11]]
    n2 = [ufloat(count, np.sqrt(count)) for count in n2]
    t2 = [t[0]+t[1], t[2]+t[3], t[4]+t[5], t[6]+t[7], t[8]+t[9], t[10]+t[11]]
    r2 = weightedAverage(np.array(n2)/np.array(t2))
    print(r2, np.array(n2)/np.array(t2))

    n3 = [n[0]+n[1]+n[2], n[3]+n[4]+n[5], n[6]+n[7]+n[8], n[9]+n[10]+n[11]]
    n3 = [ufloat(count, np.sqrt(count)) for count in n3]
    t3 = [t[0]+t[1]+t[2], t[3]+t[4]+t[5], t[6]+t[7]+t[8], t[9]+t[10]+t[11]]
    r3 = weightedAverage(np.array(n3)/np.array(t3))
    print(r3, np.array(n3)/np.array(t3))

    n4 = [n[0]+n[1]+n[2]+n[3], n[4]+n[5]+n[6]+n[7], n[8]+n[9]+n[10]+n[11]]
    n4 = [ufloat(count, np.sqrt(count)) for count in n4]
    t4 = [t[0]+t[1]+t[2]+t[3], t[4]+t[5]+t[6]+t[7], t[8]+t[9]+t[10]+t[11]]
    r4 = weightedAverage(np.array(n4)/np.array(t4))    
    print(r4, np.array(n4)/np.array(t4))

    n6 = [n[0]+n[1]+n[2]+n[3]+n[4]+n[5], n[6]+n[7]+n[8]+n[9]+n[10]+n[11]]
    n6 = [ufloat(count, np.sqrt(count)) for count in n6]
    t6 = [t[0]+t[1]+t[2]+t[3]+t[4]+t[5], t[6]+t[7]+t[8]+t[9]+t[10]+t[11]]
    r6 = weightedAverage(np.array(n6)/np.array(t6))
    print(r6, np.array(n6)/np.array(t6))

    n12 = np.sum(n)
    n12 = ufloat(n12, np.sqrt(n12))
    t12 = np.sum(t)
    r12 = n12/t12
    print(r12, np.array(n12)/np.array(t12))

    gridSpec = [1, 1]
    figSize, marginRatio = getSizeMargin(gridSpec, subplotSize=[18.0, 10.0])
    fig = plt.figure(figsize=figSize); fig.subplots_adjust(*marginRatio)
    gs = gridspec.GridSpec(*gridSpec)
    matplotlib.rc('xtick', labelsize=24)
    matplotlib.rc('ytick', labelsize=24)
    ax = []
    for axIdx in range(gridSpec[0]*gridSpec[1]):
        ax.append(fig.add_subplot(gs[axIdx]));
        ax[-1].ticklabel_format(style='sci', scilimits=(-3, 3))
        ax[-1].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[-1].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[-1].xaxis.set_tick_params(length=6, width=1, which='major')
        ax[-1].xaxis.set_tick_params(length=3, width=1, which='minor')
        ax[-1].yaxis.set_tick_params(length=6, width=1, which='major')
        ax[-1].yaxis.set_tick_params(length=3, width=1, which='minor')

    inputPars ={'markersize':10.0,'markeredgewidth':3.0,'capsize':5.0,'elinewidth':4.0}
    sumLabels = [1, 2, 3, 4, 5, 6]
    rateArr = [r1.n, r2.n, r3.n, r4.n, r6.n, r12.n]
    rateErr = [r1.s, r2.s, r3.s, r4.s, r6.s, r12.s]
    ax[0].errorbar(sumLabels, rateArr, yerr=rateErr, color='blue', fmt='o',**inputPars)
    ax[0].set_title('Q5 rate from Poisson weighted average', fontsize=36, y=1.03)
    ax[0].set_xlabel('sum label', fontsize=32)
    ax[0].set_xticklabels(['', 1, 2, 3, 4, 6, 12])
    ax[0].set_ylabel('rate (Hz)', fontsize=32)
    plt.savefig('Q5.png')
    plt.clf()
    print('saving: Q5.png')
#######################################################################################
def func_lin(t, offset, slope): return (offset + slope*t)
def weightedAverage(ufloatArr):    #correlation not considered
    valArr = np.array([meas.n for meas in ufloatArr])
    errArr = np.array([meas.s for meas in ufloatArr])
    weights = 1.0/np.power(errArr, 2)
    return ufloat(np.average(valArr, weights=weights), np.sqrt(1.0/np.sum(weights)))
DEFAULT_SCALEYX=((1.0-0.1-0.11)*7.0)/((1.0-0.13-0.08)*9.0)
def getSizeMargin(gridSpec, subplotSize=[9.0, 7.0],plotMargin=[0.13, 0.1, 0.08, 0.11]):
    figSize = (gridSpec[1]*subplotSize[0], gridSpec[0]*subplotSize[1])
    marginRatio = [plotMargin[0]/gridSpec[1], plotMargin[1]/gridSpec[0],\
                   1.0-plotMargin[2]/gridSpec[1], 1.0-plotMargin[3]/gridSpec[0],\
                   (plotMargin[0]+plotMargin[2])/(1.0-plotMargin[0]-plotMargin[2]),\
                   (plotMargin[1]+plotMargin[3])/(1.0-plotMargin[1]-plotMargin[3])]
    return figSize, marginRatio 
#######################################################################################
if __name__ == "__main__": main()




