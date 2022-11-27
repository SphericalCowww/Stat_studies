import os, sys, pathlib, time, re, glob, math
import numpy as np
import matplotlib as mpl
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
    #Q: how to get covariance given only the variable mean and std?
    # =>
    x = ufloat(2.0, 1.0)
    y = ufloat(3.0, 1.0)
    z = 3*umath.pow(x, 2) + 4*y
    #z = 3*umath.sqrt(x) + 4*y
    print(z)
    #does the following assume Gaussian uncertainty?
    #otherwise, need E(x^3) for z=3x^2+4y and
    #math.stackexchange.com/questions/547671
    print(uncertainties.correlation_matrix([x, y, z]))
    #alos interesting, see Uncorrelatedness and independence:
    #en.wikipedia.org/wiki/Covariance

    print("--------------------------------------------------------------------------")
    
    #Q: assuming y = kx, with k fixed
    #measuring x and y indepedently, and want z = x + y
    #does Var(z) = Var(x) + Var(y)?
    # =>
    x = ufloat(1.05, 0.05)
    y = ufloat(2.03, 0.05)
    k = 2.0
    z = x + y
    print(z.format("10.5f"))
    z1 = (1.0 + k)*x
    z2 = (1.0/k + 1.0)*y
    weightedErr = math.sqrt(1.0/(1.0/pow(z1.std_dev, 2) + 1.0/pow(z2.std_dev, 2)))
    weightedAve = (z1.n/pow(z1.std_dev, 2)+z2.n/pow(z2.std_dev, 2))*pow(weightedErr, 2)
    print(z1.format("10.5f"))
    print(z2.format("10.5f"))
    print("   "+str(round(weightedAve, 5))+"+/-     "+str(round(weightedErr, 5)))
    print(((z1+z2)/2.0).format("10.5f"))

    #Q: standard error of weighted average with correlation?
    #uncertainties doesn't have weighted average:
    #stackoverflow.com/questions/43637370

#######################################################################################
if __name__ == "__main__": main()




