"""
This file contains all functions used in the fitting algorithm
It first runs a fitting with 3 modes and then reduces the number of modes if some modes are overlaping
Use the function 'MAIN_FIT' to run the fitting algorithm, with:
Dp: diameter sizes,
distrib: particle number size distribution
The argument finescanning can be set to 'no' to avoid the fine scanning step and speed up the fitting
The function returns the fitted parameters for the 3 modes in the following order:
Dpg, sig, Ntot (each of these outputs the length equals to the number of modes), 

This code was originally coded in MATLAB by Tareq Hussein, September 2006

Adapted by Théodore Khadir, August 2022
Contact info: theodore.khadir@aces.su.se
"""

from scipy import interpolate
import numpy as np
import warnings
# don't show warnings
warnings.filterwarnings("ignore")

def MAIN_FIT(Dp, distrib, TOL_1N = 0.010, TOL_2N = 0.005, finescanning='yes'):
    """
    TIPS on using TOL_xN:
    TOL_1N: increasing this value --> increases the preference for 1-mode fitting
    TOL_2N: increasing this value --> increases the preference for 2-modes fitting
    TOL_3N: increasing this value --> increases the preference for 3-modes fitting

    []. Forcing the algorithm to reduce the number of modes if some modes
        are overlaping:
            TOL_1N = inf;
            TOL_2N = inf;
            TOL_3N = inf;
            TOL_4N = inf;

    []. Forcing the algorithm to keep on the number of modes selected by the
        user. For example, it will fit only 2-modes if N = 2:
            TOL_1N = -inf;
            TOL_2N = -inf;
            TOL_3N = -inf;
            TOL_4N = -inf;

    Following are suggestions and default values after evaluation to the data
    sets presented in Hussein et al. (2005 Boreal Environment Research 10:
    337ñ355). !!NOTE THAT THESE ARE NOT ABSOLUTE!!

    []. Urban and suburban aerosols:        Helsinki, Melpitz,
    []. Polar aerosols or similar sites:    Antarctica,
    []. Marine aerosols:                    Mace Head (this may need 5-modes!), Atlanta,
    []. Remote/Rural regions:               V‰rriˆ,
    []. Boreal forest or similar sites:     Hyyti‰l‰,
            TOL_1N = 0.010;
            TOL_2N = 0.005;
            TOL_3N = 0.005;
            TOL_4N = 0.005;
            
    """
    # Perform the fitting with 3 modes
    model_param = DO_FIT_400_3M(Dp, distrib, finescanning=finescanning)
    # modes 1, 2, 3 are eliminated if null concentration or less
    model_param[:3] = DO_FIT_eliminate_null(model_param[:3])
    # Reduce the number of modes if some modes are overlaping
    model_param = Reduce_3Mto2M(Dp, model_param,TOL_1N,TOL_2N,distrib)
    return model_param[:3] # doesn't return the limit and LIMIT
    
def DO_FIT_400_3M(Dp, distrib, finescanning='yes'):
    """ 
    This function returns the fitted parameters in the following order:
    Dpg: The fitted geometric mean diameters of the 3 modes (list object)
    sig: The fitted standard deviations of the 3 modes (list object)
    Ntot: The fitted total number concentration of the 3 modes (list object)
    limit: The least squares value of the fitted distribution (float object)
    LIMIT: The least squares value of the fitted distribution (float object)
    """
    
    # Dp is the particle sizes / it has to be in meters !!
    
    ## STEP 01 of the Fitting:   Searching for proper parameters by using 'DO_FIT_scan.m'
    # defining initial fitting parameters, by automatic guessing according
    # to several physical concepts and assumptions:
    N = 14                     # Number of iterations
    limit = np.inf                 # starting least squares value
    LIMIT = np.inf                 # starting least squares value
    sig = np.array([1.5,1.5,1.5]) # Starting standared deviations
    
    try:
        # Following is constrained particle size ranges of each mode...
        # You can change these values to fit your data according to your knowledge on the PNSDs you are using
        Dp_min = np.array([2, 35, 90])*10**(-9)
        Dp_max = np.array([35, 90, 400])*10**(-9)
        
        Dpg,sig,Ntot,limit,LIMIT,dlogDpg = Dpg_scan_3M(distrib, Dp, N, limit, LIMIT, sig, Dp_min, Dp_max)
        sig,Ntot,limit,LIMIT = sig_scan_3M(Dpg,Ntot,N,limit,LIMIT,Dp,distrib, sig)
    except:
        # The matrix is often non invertible and changing the min Dp can solve the issue
        # print('exception in DO_FIT_400_3M, trying to change Dp_min')
        Dp_min = np.array([2, 34, 89])*10**(-9)
        Dp_max = np.array([35, 90, 400])*10**(-9)
        
        Dpg,sig,Ntot,limit,LIMIT,dlogDpg = Dpg_scan_3M(distrib, Dp, N, limit, LIMIT, sig, Dp_min, Dp_max)
        sig,Ntot,limit,LIMIT = sig_scan_3M(Dpg,Ntot,N,limit,LIMIT,Dp,distrib, sig)
    if finescanning == 'yes':
        # STEP 02 of the Fitting:   Fine fitting the particle size distribution by using "DO_FIT_fix()" several times
        N = 18
        for jj in range(0,2): # double scan
            Dp_min = 10**(np.log10(Dpg) - np.array([6,3,1]) * dlogDpg)
            Dp_max = 10**(np.log10(Dpg) + np.array([4,4,6]) * dlogDpg)

            Dpg,sig,Ntot,limit,LIMIT = DO_Fit_FIX(distrib,Dp,N,limit,LIMIT,Dpg,sig,Ntot,Dp_min,Dp_max)
        
    return [Dpg, sig, Ntot, limit,LIMIT]
        
def f_lognorm(Dpg,sig,Dp):
    """
    This function returns the lognormal distribution of the given parameters for the given particle sizes
    """
    return 0.39894228 / np.log10(sig) * np.exp(-0.5*(np.log10(Dp)-np.log10(Dpg))**2/(np.log10(sig)**2))

def Dpg_scan_3M(distrib, Dp, N, limit_, LIMIT_, sig_, Dp_min, Dp_max):
    """
    Finds the best fitting Dpg by scanning the geometric mean diameters of the modes
    and minimizing the least squares value
    This function returns the fitted parameters in the following order:
    Dpg: The fitted geometric mean diameters of the 3 modes (list object)
    sig: The fitted standard deviations of the 3 modes (list object)
    Ntot: The fitted total number concentration of the 3 modes (list object)
    limit: The least squares value of the fitted distribution (float object)
    LIMIT: The least squares value of the fitted distribution (float object)
    """
    dlogDpg = np.zeros(3)*np.nan
    # mode 1
    Dpg_1 = np.logspace(np.log10(Dp_max[0]), np.log10(Dp_min[0]), N, base=10.0, endpoint=True)
    dlogDpg[0] = np.log10(Dpg_1[1])-np.log10(Dpg_1[0])
    # mode 2
    Dpg_2 = np.logspace(np.log10(Dp_max[1]), np.log10(Dp_min[1]), N, base=10.0, endpoint=True)
    dlogDpg[1] = np.log10(Dpg_2[1])-np.log10(Dpg_2[0])
    # mode 3
    Dpg_3 = np.logspace(np.log10(Dp_max[2]), np.log10(Dp_min[2]), N, base=10.0, endpoint=True)
    dlogDpg[2] = np.log10(Dpg_3[1])-np.log10(Dpg_3[0])
    
    limit = limit_
    LIMIT = LIMIT_
    
    A = np.zeros((3,3))*np.nan
    F = np.zeros((3,1))*np.nan
    
    for i1 in range(0,N):
        A1 = f_lognorm(Dpg_1[i1], sig_[0], Dp)
        A[0,0] = np.sum(A1*A1)
        F[0,0] = np.sum(distrib*A1)
        
        for i2 in range(0,N):
            A2 = f_lognorm(Dpg_2[i2], sig_[1], Dp)
            A[1,1] = np.sum(A2*A2)
            A[1,0] = np.sum(A2*A1)
            A[0,1] = A[1,0]
            F[1,0] = np.sum(distrib*A2)
            
            for i3 in range(0,N):
                A3 = f_lognorm(Dpg_3[i3], sig_[2], Dp)
                A[2,2] = np.sum(A3*A3)
                A[2,0] = np.sum(A3*A1)
                A[0,2] = A[2,0]
                A[2,1] = np.sum(A3*A2)
                A[1,2] = A[2,1]
                F[2,0] = np.sum(distrib*A3)
                
                Ntot_ = (np.linalg.inv(A).dot(F))

                if ((Ntot_[0]>=0) & (Ntot_[1]>=0) & (Ntot_[2]>=0)):
                    fitting  = (Ntot_[0] * A1 + Ntot_[1] * A2 + Ntot_[2] * A3) - distrib
                    variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))
                    
                    FITTING  = np.log10((Ntot_[0] * A1 + Ntot_[1] * A2 + Ntot_[2] * A3) / distrib)
                    VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

                    if ((variance <= limit)):
                        limit = variance
                        LIMIT = VARIANCE
                        Dpg = [Dpg_1[i1],Dpg_2[i2],Dpg_3[i3]]
                        Ntot = Ntot_
                        sig  = sig_
                               
    # if didnt improve the least mean squares, returns previous set of sig
    # here don't perform the finescanning (not ready yet but not too needed)
    if ((limit == limit_) or (LIMIT == LIMIT_)):
        sig = sig_
        Ntot = Ntot_
#         [Dpg,sig,Ntot,limit,LIMIT,dlogDpg] = DO_FIT_400_finescanDpg(Dp,distrib,sig_,Dp_min,Dp_max,3)
                               
    return [Dpg,sig,Ntot,limit,LIMIT,dlogDpg]

def sig_scan_3M(Dpg_,Ntot_,N,limit_,LIMIT_, Dp,distrib, sig_):
    """
    Finds the best fitting sig by scanning the standard deviations of the modes
    and minimizing the least squares value
    This function returns the fitted parameters in the following order:
    sig: The fitted standard deviations of the 3 modes (list object)
    Ntot: The fitted total number concentration of the 3 modes (list object)
    limit: The least squares value of the fitted distribution (float object)
    LIMIT: The least squares value of the fitted distribution (float object)
    """
    sig_min = 1.1
    sig_max = 2.1
    dsig    = (sig_max - sig_min) / N
    sig_1 = np.arange(sig_min, sig_max+dsig, dsig)
    sig_2 = sig_1
    sig_3 = sig_1
    
    limit = limit_
    LIMIT = LIMIT_
    
    A = np.zeros((3,3))*np.nan
    F = np.zeros((3,1))*np.nan
    
    for i1 in range(0,N):
        A1 = f_lognorm(Dpg_[0], sig_1[i1], Dp)
        A[0,0] = np.sum(A1*A1)
        F[0,0] = np.sum(distrib*A1)
        
        for i2 in range(0,N):
            A2 = f_lognorm(Dpg_[1], sig_2[i2], Dp)
            A[1,1] = np.sum(A2*A2)
            A[1,0] = np.sum(A2*A1)
            A[0,1] = A[1,0]
            F[1,0] = np.sum(distrib*A2)
            
            for i3 in range(0,N):
                A3 = f_lognorm(Dpg_[2], sig_3[i3], Dp)
                A[2,2] = np.sum(A3*A3)
                A[2,0] = np.sum(A3*A1)
                A[0,2] = A[2,0]
                A[2,1] = np.sum(A3*A2)
                A[1,2] = A[2,1]
                F[2,0] = np.sum(distrib*A3)
                
                Ntot__ = (np.linalg.inv(A).dot(F))
                if ((Ntot__[0]>=0) & (Ntot__[1]>=0) & (Ntot__[2]>=0)):
                    fitting  = (Ntot__[0] * A1 + Ntot__[1] * A2 + Ntot__[2] * A3) - distrib
                    variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

                    FITTING  = np.log10((Ntot__[0] * A1 + Ntot__[1] * A2 + Ntot__[2] * A3) / distrib)
                    VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

                    if ((variance <= limit)):
                        limit = variance
                        LIMIT = VARIANCE
                        sig = [sig_1[i1],sig_2[i2],sig_3[i3]]
                        Ntot = Ntot__                

    if ((limit == limit_) or (LIMIT == LIMIT_)):
        sig = sig_
        Ntot = Ntot_
    
    return [sig,Ntot,limit,LIMIT]

def DO_Fit_FIX(distrib,Dp,N,limit,LIMIT,Dpg,sig,Ntot,Dp_min,Dp_max):
    Dpg_1 = np.logspace(np.log10(Dp_max[0]), np.log10(Dp_min[0]), N, base=10.0, endpoint=True)
    Dpg_2 = np.logspace(np.log10(Dp_max[1]), np.log10(Dp_min[1]), N, base=10.0, endpoint=True)
    Dpg_3 = np.logspace(np.log10(Dp_max[2]), np.log10(Dp_min[2]), N, base=10.0, endpoint=True)
    
    sig_min = 1.1
    sig_max = 2.1
    dsig    = (sig_max - sig_min) / N
    sig_1 = np.arange(sig_min, sig_max+dsig, dsig)
    sig_2 = sig_1
    sig_3 = sig_1
    
    [Dpg,Ntot,limit,LIMIT] = Iterate_Dpg_FIX(Dpg,sig,Ntot,N,limit,LIMIT,Dp,distrib, Dpg_1,Dpg_2,Dpg_3)
    [sig,Ntot,limit,LIMIT] = Iterate_sig_FIX(Dpg,sig,Ntot,N,limit,LIMIT,Dp,distrib, sig_1,sig_2,sig_3)
    [Dpg,sig,Ntot,limit,LIMIT] = FIX_accu(Dp,distrib,Dpg,sig,Ntot,limit,LIMIT)
    [Dpg,sig,Ntot,limit,LIMIT] = FIX_aitk(Dp,distrib,Dpg,sig,Ntot,limit,LIMIT)
    [Dpg,sig,Ntot,limit,LIMIT] = FIX_nucl(Dp,distrib,Dpg,sig,Ntot,limit,LIMIT)
    
    return [Dpg,sig,Ntot,limit,LIMIT]
    
def Iterate_Dpg_FIX(Dpg_,sig_,Ntot_,N,limit_,LIMIT_,Dp,distrib, Dpg_1,Dpg_2,Dpg_3):
    limit = limit_
    LIMIT = LIMIT_
    
    A = np.zeros((3,3))*np.nan
    F = np.zeros((3,1))*np.nan
    
    for i1 in range(0,N):
        A1 = f_lognorm(Dpg_1[i1], sig_[0], Dp)
        A[0,0] = np.sum(A1*A1)
        F[0,0] = np.sum(distrib*A1)
        
        for i2 in range(0,N):
            A2 = f_lognorm(Dpg_2[i2], sig_[1], Dp)
            A[1,1] = np.sum(A2*A2)
            A[1,0] = np.sum(A2*A1)
            A[0,1] = A[1,0]
            F[1,0] = np.sum(distrib*A2)
            
            for i3 in range(0,N):
                A3 = f_lognorm(Dpg_3[i3], sig_[2], Dp)
                A[2,2] = np.sum(A3*A3)
                A[2,0] = np.sum(A3*A1)
                A[0,2] = A[2,0]
                A[2,1] = np.sum(A3*A2)
                A[1,2] = A[2,1]
                F[2,0] = np.sum(distrib*A3)
                
                Ntot__ = (np.linalg.inv(A).dot(F))

                if ((Ntot__[0]>=0) & (Ntot__[1]>=0) & (Ntot__[2]>=0)):
                    fitting  = (Ntot__[0] * A1 + Ntot__[1] * A2 + Ntot__[2] * A3) - distrib
                    variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

                    FITTING  = np.log10((Ntot__[0] * A1 + Ntot__[1] * A2 + Ntot__[2] * A3) / distrib)
                    VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

                    if ((variance <= limit)):
                        limit = variance
                        LIMIT = VARIANCE
                        Dpg = [Dpg_1[i1],Dpg_2[i2],Dpg_3[i3]]
                        Ntot = Ntot__
                        sig  = sig_
             
    if ((limit == limit_) or (LIMIT == LIMIT_)):
        Dpg = Dpg_
        Ntot = Ntot_
                               
    return [Dpg,Ntot,limit,LIMIT]

def Iterate_sig_FIX(Dpg_,sig_,Ntot_,N,limit_,LIMIT_, Dp,distrib, sig_1,sig_2,sig_3):
    A = np.zeros((3,3))*np.nan
    F = np.zeros((3,1))*np.nan
    
    limit = limit_
    LIMIT = LIMIT_
    
    for i1 in range(0,N):
        A1 = f_lognorm(Dpg_[0], sig_1[i1], Dp)
        A[0,0] = np.sum(A1*A1)
        F[0,0] = np.sum(distrib*A1)
        
        for i2 in range(0,N):
            A2 = f_lognorm(Dpg_[1], sig_2[i2], Dp)
            A[1,1] = np.sum(A2*A2)
            A[1,0] = np.sum(A2*A1)
            A[0,1] = A[1,0]
            F[1,0] = np.sum(distrib*A2)
            
            for i3 in range(0,N):
                A3 = f_lognorm(Dpg_[2], sig_3[i3], Dp)
                A[2,2] = np.sum(A3*A3)
                A[2,0] = np.sum(A3*A1)
                A[0,2] = A[2,0]
                A[2,1] = np.sum(A3*A2)
                A[1,2] = A[2,1]
                F[2,0] = np.sum(distrib*A3)
                
                Ntot__ = (np.linalg.inv(A).dot(F))
                if ((Ntot__[0]>=0) & (Ntot__[1]>=0) & (Ntot__[2]>=0)):
                    fitting  = (Ntot__[0] * A1 + Ntot__[1] * A2 + Ntot__[2] * A3) - distrib
                    variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

                    FITTING  = np.log10((Ntot__[0] * A1 + Ntot__[1] * A2 + Ntot__[2] * A3) / distrib)
                    VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

                    if ((variance <= limit)):
                        limit = variance
                        LIMIT = VARIANCE
                        sig = [sig_1[i1],sig_2[i2],sig_3[i3]]
                        Ntot = Ntot__
                        
    if ((limit == limit_) or (LIMIT == LIMIT_)):
        sig = sig_
        Ntot = Ntot_
    
    return [sig,Ntot,limit,LIMIT]

def FIX_accu(Dp,distrib,Dpg_,sig_,Ntot_,limit_,LIMIT_):
    A = np.zeros((3,3))*np.nan
    F = np.zeros((3,1))*np.nan
    
    N = 18
    Dpg_3 = np.logspace(np.log10(150*(10**(-9))), np.log10(1000*(10**(-9))), N, base=10.0, endpoint=True)
    sig_min = 1.1
    sig_max = 2.1
    dsig    = (sig_max - sig_min) / N
    sig_3 = np.arange(sig_min, sig_max+dsig, dsig)
    limit = limit_
    LIMIT = LIMIT_
    
    A1 = f_lognorm(Dpg_[0], sig_[0], Dp)
    A[0,0] = np.sum(A1*A1)
    F[0,0] = np.sum(distrib*A1)
        
    A2 = f_lognorm(Dpg_[1], sig_[1], Dp)
    A[1,1] = np.sum(A2*A2)
    A[1,0] = np.sum(A2*A1)
    A[0,1] = A[1,0]
    F[1,0] = np.sum(distrib*A2)
    
    for ii in range(0,N):
        for jj in range(0,N):
            A3 = f_lognorm(Dpg_3[ii], sig_3[jj], Dp)
            A[2,2] = np.sum(A3*A3)
            A[2,0] = np.sum(A3*A1)
            A[0,2] = A[2,0]
            A[2,1] = np.sum(A3*A2)
            A[1,2] = A[2,1]
            F[2,0] = np.sum(distrib*A3)
            
            Ntot__ = (np.linalg.inv(A).dot(F))
            if ((Ntot__[0]>=0) & (Ntot__[1]>=0) & (Ntot__[2]>=0)):
                fitting  = (Ntot__[0] * A1 + Ntot__[1] * A2 + Ntot__[2] * A3) - distrib
                variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

                FITTING  = np.log10((Ntot__[0] * A1 + Ntot__[1] * A2 + Ntot__[2] * A3) / distrib)
                VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

                if ((variance <= limit)):
                    limit = variance
                    LIMIT = VARIANCE
                    Ntot = Ntot__
                    Dpg = [Dpg_[0], Dpg_[1], Dpg_3[ii]]
                    sig = [sig_[0], sig_[1], sig_3[jj]]
                    
    if ((limit == limit_) or (LIMIT == LIMIT_)):
        sig = sig_
        Ntot = Ntot_
        Dpg = Dpg_
        
    return [Dpg,sig,Ntot,limit,LIMIT]

def FIX_aitk(Dp,distrib,Dpg_,sig_,Ntot_,limit_,LIMIT_):
    A = np.zeros((3,3))*np.nan
    F = np.zeros((3,1))*np.nan
    N = 18
    Dpg_2 = np.logspace(np.log10(25*(10**(-9))), np.log10(90*(10**(-9))), N, base=10.0, endpoint=True)
    sig_min = 1.1
    sig_max = 2.1
    dsig    = (sig_max - sig_min) / N
    sig_2 = np.arange(sig_min, sig_max+dsig, dsig)
    limit = limit_
    LIMIT = LIMIT_
    
    A1 = f_lognorm(Dpg_[0], sig_[0], Dp)
    A[0,0] = np.sum(A1*A1)
    F[0,0] = np.sum(distrib*A1)
        
    A3 = f_lognorm(Dpg_[2], sig_[2], Dp)
    A[2,2] = np.sum(A3*A3)
    A[2,0] = np.sum(A3*A1)
    A[0,2] = A[2,0]
    F[2,0] = np.sum(distrib*A3)
    
    for ii in range(0,N):
        for jj in range(0,N):
            A2 = f_lognorm(Dpg_2[ii], sig_2[jj], Dp)
            A[1,1] = np.sum(A2*A2)
            A[1,0] = np.sum(A2*A1)
            A[0,1] = A[1,0]
            A[2,1] = np.sum(A3*A2)
            A[1,2] = A[2,1]
            F[1,0] = np.sum(distrib*A2)
            
            Ntot__ = (np.linalg.inv(A).dot(F))
            if ((Ntot__[0]>=0) & (Ntot__[1]>=0) & (Ntot__[2]>=0)):
                fitting  = (Ntot__[0] * A1 + Ntot__[1] * A2 + Ntot__[2] * A3) - distrib
                variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

                FITTING  = np.log10((Ntot__[0] * A1 + Ntot__[1] * A2 + Ntot__[2] * A3) / distrib)
                VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

                if ((variance <= limit)):
                    limit = variance
                    LIMIT = VARIANCE
                    Ntot = Ntot__
                    Dpg = [Dpg_[0], Dpg_2[ii], Dpg_[2]]
                    sig = [sig_[0], sig_2[jj], sig_[2]]
                    
    if ((limit == limit_) or (LIMIT == LIMIT_)):
        sig = sig_
        Ntot = Ntot_
        Dpg = Dpg_
        
    return [Dpg,sig,Ntot,limit,LIMIT]

def FIX_nucl(Dp,distrib,Dpg_,sig_,Ntot_,limit_,LIMIT_):
    A = np.zeros((3,3))*np.nan
    F = np.zeros((3,1))*np.nan
    N = 18
    Dpg_1 = np.logspace(np.log10(16*(10**(-9))), np.log10(2*(10**(-9))), N, base=10.0, endpoint=True)
    sig_min = 1.1
    sig_max = 2.0
    dsig    = (sig_max - sig_min) / N
    sig_1 = np.arange(sig_min, sig_max+dsig, dsig)
    limit = limit_
    LIMIT = LIMIT_
    
    A3 = f_lognorm(Dpg_[2], sig_[2], Dp)
    A[2,2] = np.sum(A3*A3)
    F[2,0] = np.sum(distrib*A3)
    
    A2 = f_lognorm(Dpg_[1], sig_[1], Dp)
    A[1,1] = np.sum(A2*A2)
    A[1,2] = np.sum(A2*A3)
    A[2,1] = A[1,2]
    F[1,0] = np.sum(distrib*A2)
    
    for ii in range(0,N):
        for jj in range(0,N):
            A1 = f_lognorm(Dpg_1[ii], sig_1[jj], Dp)
            A[0,0] = np.sum(A1*A1)
            A[1,0] = np.sum(A2*A1)
            A[0,1] = A[1,0]
            A[2,0] = np.sum(A3*A1)
            A[0,2] = A[2,0]
            F[1,0] = np.sum(distrib*A1)
            
            Ntot__ = (np.linalg.inv(A).dot(F))
            if ((Ntot__[0]>=0) & (Ntot__[1]>=0) & (Ntot__[2]>=0)):
                fitting  = (Ntot__[0] * A1 + Ntot__[1] * A2 + Ntot__[2] * A3) - distrib
                variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

                FITTING  = np.log10((Ntot__[0] * A1 + Ntot__[1] * A2 + Ntot__[2] * A3) / distrib)
                VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

                if ((variance <= limit)):
                    limit = variance
                    LIMIT = VARIANCE
                    Ntot = Ntot__
                    Dpg = [Dpg_[0], Dpg_2[ii], Dpg_[2]]
                    sig = [sig_[0], sig_2[jj], sig_[2]]
                    
    if ((limit == limit_) or (LIMIT == LIMIT_)):
        sig = sig_
        Ntot = Ntot_
        Dpg = Dpg_
        
    return [Dpg,sig,Ntot,limit,LIMIT]

def DO_FIT_eliminate_null(model_param):
    #Check if any of the modes consists of zero concentration or less
    
    # mode 1, 2, 3 are eliminated if null concentration or less
    if ((model_param[2][0] == 0) or (model_param[2][0] < 0)):
        model_param[2][0] = np.nan
    if ((model_param[2][1] == 0) or (model_param[2][1] < 0)):
        model_param[2][1] = np.nan
    if ((model_param[2][2] == 0) or (model_param[2][2] < 0)):
        model_param[2][2] = np.nan
    
    return model_param

def Reduce_3Mto2M(Dp, model_param3M,TOL_1N,TOL_2N,distrib):
    #-------------------------------------------------------------------------%
    # Checks if it is possible to reduce 3 modes to 2 modes
    #-------------------------------------------------------------------------%
    
    model_param = model_param3M
    
    Dpg = model_param3M[0]
    sig = model_param3M[1]
    N = model_param3M[2]
    
    condition_12 = DO_FIT_400_overlap(Dpg[0],sig[0],N[0],Dpg[1],sig[1],N[1])
    condition_23 = DO_FIT_400_overlap(Dpg[1],sig[1],N[1],Dpg[2],sig[2],N[2])
    
    if ((condition_12 == 1) or (condition_23 == 1)):

        # print('   Some modes seem to be overlapping! Proceeding with 2-modes fittings ...')

        model_param2NA = DO_FIT_400_2M_NA_(Dp, distrib)
        model_param2NN = DO_FIT_400_2M_NN_(Dp, distrib)

        if (model_param2NA[3] <= model_param2NN[3]):
            model_param2M = model_param2NA
        else:
            model_param2M = model_param2NN

        # Deciding between 2 and 3 modes:
        if model_param2M[3] <= model_param3M[3] * (1 + TOL_2N):
            # print('      2-modes fitting is better.')
            model_param = Reduce_2Mto1M(Dp, model_param2M,TOL_1N,distrib)
        else:
            # print('      3-modes fitting is ok.')
            model_param = model_param3M

    else:
        # print('   No modes overlapping.')
        model_param = model_param3M
    
    return model_param

def Reduce_2Mto1M(Dp, model_param2M,TOL_1N,distrib):
    #-------------------------------------------------------------------------%
    # Checks if possible to reduce 2 modes to 1 modes
    #-------------------------------------------------------------------------%
    
    model_param = model_param2M
    
    Dpg = model_param2M[0]
    sig = model_param2M[1]
    N = model_param2M[2]
    
    condition_12 = DO_FIT_400_overlap(Dpg[0],sig[0],N[0],Dpg[1],sig[1],N[1])
    
    if (condition_12 == 1):
        # print('   Some modes seem to be overlapping! Proceeding with 1-modes fittings ...')
        model_param1M = DO_FIT_400_1M(Dp, distrib)

        if (model_param1M[3] <= model_param2M[3]*(1 + TOL_1N)):
            model_param = model_param1M
        else:
            model_param = model_param2M
            
    else:
        # print('   No modes overlapping.')
        model_param = model_param2M
    
    return model_param
    
    
def DO_FIT_400_overlap(Dpg1,sig1,N1,Dpg2,sig2,N2):
    #-------------------------------------------------------------------------%
    # This subroutine tests the overlapping condition between any two adjacent
    # modes. It considers the mode with bigger variance (sig1) as the central
    # mode and then finds out the allowed minimum concentration of the other
    # mode based on the ratios sig2/sig1 and |log10(Dpg2/Dpg1)|.
    #
    # The subroutine returns 0/1 for not-overlapping/overlapping.
    #
    # Hypothesis of overlapping:
    #
    # Any two adjacent modes are allowed to co-exist around each others only if
    # they can not be replaced with a single mode that does not change the
    # overall fitting quality. In that sense, two main point are taken into
    # account:
    #   1. How much does the minor mode deforms the central mode? For example,
    #      a mode with very small "sig" would easily deforms the central mode
    #      and thus they do not overlap. On the other hand, two adjacent modes
    #      with rather similar sig's can overlap easily, and thus can be
    #      repplaced with one mode.
    #   2. There is a certain distance between two adjacent modes to co-exist
    #      freely without any restrictions. Based on my own observation, this
    #      is according to |log10(Dpg2/Dpg1)| = 0.83276 for all sig's. However,
    #      as sig2/sig1 becomes larger than ~0.95 or smaller than ~0.65, this
    #      distance becomes shorter.
    #
    # Note that these findings regarding the mode overlapping may vary from one
    # person to another. However, using one hypothesis for the whole data-set
    # or different data-sets, to be compared with each others, allow reduction
    # for the user influence.
    #
    # Due to the expensive computations, the fitting procedure is performed
    # with an optimal number of iterations. That may, in other words, influence
    # the fitting quality as a whole. However, allowing for more iterations
    # with and using faster computers may provide better fitting quality.
    # 
    #
    # Tareq Hussein
    # August 2006, Stockholm.

    # Re-arrangements of the modes to select the central mode with bigger sig
    if sig1 > sig2:
        sig = [sig1,sig2]
        Dpg = [Dpg1,Dpg2]
        N   = [N1,N2]
    else:
        sig = [sig2,sig1]
        Dpg = [Dpg2,Dpg1]
        N   = [N2,N1]
        
    # This is the allowed minimum concentration ratio N2/N1 (x100) that defines
    # overlapping between adjacent modes with respect to their relative
    # variances and geometric mean diamters.
    N21_min = np.array([
                [np.nan  ,       1.00,0.95,0.90,0.80 ,  0.75,0.70,0.65  ,  0.60  ,  0.55  ,  0.52],
                [0.03620  ,   np.nan   ,  np.nan  ,   np.nan  ,   np.nan  ,   30   ,   10     , 5     ,  3  ,     1 ,      0],
                [0.10862  ,  np.nan ,   np.nan   ,  60   ,   40  ,    30   ,   10  ,    6   ,    3    ,   1   ,    0],
                [0.18103  ,   np.nan ,   np.nan  ,   80   ,   35  ,    20   ,   10  ,    7   ,    3    ,   2   ,    0],
                [0.25345  ,   np.nan  ,   np.nan   ,  90   ,   30    ,  15   ,   9    ,   7   ,    4   ,    2     ,  0],
                [0.32586  ,   np.nan  ,  np.nan ,   80   ,   25   ,   15   ,   7   ,    7   ,    4   ,    2    ,  0],
                [0.39828  ,   8    ,   10   ,   30    ,  20   ,   15   ,   7    ,   7   ,    3    ,   2    ,   0],
                [0.47069 ,    3    ,   5   ,    10   ,   10   ,   8    ,   5    ,   5    ,   3    ,   2    ,   0],
                [0.54310  ,   2   ,   3    ,   5     ,  8    ,   6    ,   5    ,   5    ,   2    ,   1    ,   0],
                [0.61552   ,  1   ,    2   ,    3    ,   5    ,   5    ,   5    ,   3   ,    2    ,   1     ,  0],
                [0.68793  ,   0   ,    1    ,   2    ,   3    ,   3    ,   3    ,   2   ,    1    ,  0     ,  0],
                [0.76034  ,   0    ,   0    ,   1    ,   2   ,    2    ,   2    ,   1   ,    0    ,   0    ,   0],
                [0.83276 ,    0  ,    0   ,    0   ,    1    ,   1    ,   1   ,    0    ,   0    ,   0   ,    0],
                [0.90517  ,   0    ,   0    ,   0   ,    0    ,   0    ,   0   ,    0   ,    0    ,   0     ,  0],
                [0.97759  ,   0   ,    0   ,    0   ,    0     ,  0    ,   0    ,   0   ,    0    ,   0    ,   0],
                [1.05000  ,   0    ,   0    ,   0   ,    0   ,    0    ,   0    ,   0   ,    0     ,  0     ,  0]       ])
    
    sig21   = sig[1] / sig[0]
    N21     = N[1] / N[0]
    logD21  = np.absolute(np.log10(Dpg[1]/Dpg[0]))

    X = N21_min[0,1:]
    Y = N21_min[1:,0]
    V = N21_min[1:,1:]/100
    f = interpolate.interp2d(X, Y, V, kind='linear')
    constrain = f(sig21, logD21)

    if logD21 < N21_min[1,0]:
        condition = 1
    elif ((sig21>=N21_min[0,N21_min.shape[1]-1]) & (sig21<=N21_min[0,1]) & (logD21>=N21_min[1,0]) & (logD21<=N21_min[N21_min.shape[0]-1,0])):
        if np.isnan(constrain).sum == 0:
            if N21 >= constrain:
                condition = 0
            else:
                condition = 1
        else:
            condition = 1
    elif logD21 > N21_min[N21_min.shape[0]-1,0]:
        condition = 0
    else:
        condition = 1
        
    return condition

########################################### FIT 2 modes NA #######################################
def DO_FIT_400_2M_NA_(Dp, distrib):
    # Dp is the particle sizes from the first row of the mtrx / it has to be in meters !!
    
    ## STEP 01 of the Fitting:   Searching for proper parameters by using 'DO_FIT_scan.m'
    # defining initial fitting parameters, by automatic guessing according
    # to several physical concepts and assumptions:
    N = 18                     # Number of iterations
    limit = np.inf                 # starting least squares value
    LIMIT = np.inf                 # starting least squares value
    sig = np.array([1.75,1.75]) # Starting standared deviations
    
    # Following is constrained particle size ranges of each mode...
    Dp_min = np.array([9, 99])*10**(-9)
    Dp_max = np.array([100, 1000])*10**(-9)

    Dpg,sig,Ntot,limit,LIMIT,dlogDpg = Dpg_scan_2M(distrib, Dp, N, limit, LIMIT, sig, Dp_min, Dp_max)
    sig,Ntot,limit,LIMIT = sig_scan_2M(Dpg,Ntot,N,limit,LIMIT,Dp,distrib, sig)
    
    N = 30
    for jj in range(0,2): # double scan ?
        Dp_min = 10**(np.log10(Dpg) - np.array([3,6]) * dlogDpg)
        Dp_max = 10**(np.log10(Dpg) + np.array([6,3]) * dlogDpg)

        Dpg,sig,Ntot,limit,LIMIT = DO_Fit_FIX_2M(distrib,Dp,N,limit,LIMIT,Dpg,sig,Ntot,Dp_min,Dp_max)
    
    # limit linear least squares
    # LIMIT linear least sauares
    return [Dpg, sig, Ntot, limit, LIMIT]
        
########################################### FIT 2 modes NN #######################################
def DO_FIT_400_2M_NN_(Dp, distrib):
    # Dp is the particle sizes from the first row of the mtrx / it has to be in meters !!
    
    ## STEP 01 of the Fitting:   Searching for proper parameters by using 'DO_FIT_scan.m'
    # defining initial fitting parameters, by automatic guessing according
    # to several physical concepts and assumptions:
    N = 18                     # Number of iterations
    limit = np.inf                 # starting least squares value
    LIMIT = np.inf                 # starting least squares value
    sig = np.array([1.75,1.75]) # Starting standared deviations
    
    # Following is constrained particle size ranges of each mode...
    Dp_min = np.array([5, 74])*10**(-9)
    Dp_max = np.array([75, 200])*10**(-9)

    Dpg,sig,Ntot,limit,LIMIT,dlogDpg = Dpg_scan_2M(distrib, Dp, N, limit, LIMIT, sig, Dp_min, Dp_max)
    sig,Ntot,limit,LIMIT = sig_scan_2M(Dpg,Ntot,N,limit,LIMIT,Dp,distrib, sig)
    
    N = 30
    for jj in range(0,2): # double scan ?
        Dp_min = 10**(np.log10(Dpg) - np.array([3,6]) * dlogDpg)
        Dp_max = 10**(np.log10(Dpg) + np.array([6,3]) * dlogDpg)

        Dpg,sig,Ntot,limit,LIMIT = DO_Fit_FIX_2M(distrib,Dp,N,limit,LIMIT,Dpg,sig,Ntot,Dp_min,Dp_max)
        
    return [Dpg, sig, Ntot, limit, LIMIT]

def f_lognorm(Dpg,sig,Dp):
    return 0.39894228 / np.log10(sig) * np.exp(-0.5*(np.log10(Dp)-np.log10(Dpg))**2/(np.log10(sig)**2))


def Dpg_scan_2M(distrib, Dp, N, limit_, LIMIT_, sig_, Dp_min, Dp_max):
    
    dlogDpg = np.zeros(2)*np.nan
    # mode 1
    Dpg_1 = np.logspace(np.log10(Dp_max[0]), np.log10(Dp_min[0]), N, base=10.0, endpoint=True)
    dlogDpg[0] = np.log10(Dpg_1[1])-np.log10(Dpg_1[0])
    # mode 2
    Dpg_2 = np.logspace(np.log10(Dp_max[1]), np.log10(Dp_min[1]), N, base=10.0, endpoint=True)
    dlogDpg[1] = np.log10(Dpg_2[1])-np.log10(Dpg_2[0])
    
    limit = limit_
    LIMIT = LIMIT_
    
    A = np.zeros((2,2))*np.nan
    F = np.zeros((2,1))*np.nan
    
    for i1 in range(0,N):
        A1 = f_lognorm(Dpg_1[i1], sig_[0], Dp)
        A[0,0] = np.sum(A1*A1)
        F[0,0] = np.sum(distrib*A1)
        
        for i2 in range(0,N):
            A2 = f_lognorm(Dpg_2[i2], sig_[1], Dp)
            A[1,1] = np.sum(A2*A2)
            A[1,0] = np.sum(A2*A1)
            A[0,1] = A[1,0]
            F[1,0] = np.sum(distrib*A2)
                
            Ntot__ = (np.linalg.inv(A).dot(F))

            if ((Ntot__[0]>=0) & (Ntot__[1]>=0)):
                fitting  = (Ntot__[0] * A1 + Ntot__[1] * A2) - distrib
                variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

                FITTING  = np.log10((Ntot__[0] * A1 + Ntot__[1] * A2) / distrib)
                VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

                if ((variance <= limit)):
                    limit = variance
                    LIMIT = VARIANCE
                    Dpg = [Dpg_1[i1],Dpg_2[i2]]
                    Ntot = Ntot__
                    sig  = sig_
                               
#   #if didnt improve the least mean squares, and run a finescanning
    if ((limit == limit_) or (LIMIT == LIMIT_)):
        sig = sig_
                               
    return [Dpg,sig,Ntot,limit,LIMIT,dlogDpg]

def sig_scan_2M(Dpg_,Ntot_,N,limit_,LIMIT_, Dp,distrib, sig_):
    sig_min = 1.1
    sig_max = 2.1
    dsig    = (sig_max - sig_min) / N
    sig_1 = np.arange(sig_min, sig_max+dsig, dsig)
    sig_2 = sig_1
    
    limit = limit_
    LIMIT = LIMIT_
    
    A = np.zeros((2,2))*np.nan
    F = np.zeros((2,1))*np.nan
    
    for i1 in range(0,N):
        A1 = f_lognorm(Dpg_[0], sig_1[i1], Dp)
        A[0,0] = np.sum(A1*A1)
        F[0,0] = np.sum(distrib*A1)
        
        for i2 in range(0,N):
            A2 = f_lognorm(Dpg_[1], sig_2[i2], Dp)
            A[1,1] = np.sum(A2*A2)
            A[1,0] = np.sum(A2*A1)
            A[0,1] = A[1,0]
            F[1,0] = np.sum(distrib*A2)
                
            Ntot__ = (np.linalg.inv(A).dot(F))
            if ((Ntot__[0]>=0) & (Ntot__[1]>=0)):
                fitting  = (Ntot__[0] * A1 + Ntot__[1] * A2) - distrib
                variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

                FITTING  = np.log10((Ntot__[0] * A1 + Ntot__[1] * A2) / distrib)
                VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

                if ((variance <= limit) & (VARIANCE <= LIMIT)):
                    limit = variance
                    LIMIT = VARIANCE
                    sig = [sig_1[i1],sig_2[i2]]
                    Ntot = Ntot__                

    if ((limit == limit_) or (LIMIT == LIMIT_)):
        sig = sig_
        Ntot = Ntot_
    
    return [sig,Ntot,limit,LIMIT]

def DO_Fit_FIX_2M(distrib,Dp,N,limit,LIMIT,Dpg,sig,Ntot,Dp_min,Dp_max):
    Dpg_1 = np.logspace(np.log10(Dp_max[0]), np.log10(Dp_min[0]), N, base=10.0, endpoint=True)
    Dpg_2 = np.logspace(np.log10(Dp_max[1]), np.log10(Dp_min[1]), N, base=10.0, endpoint=True)
    
    sig_min = 1.1
    sig_max = 2.1
    dsig    = (sig_max - sig_min) / N
    sig_1 = np.arange(sig_min, sig_max+dsig, dsig)
    sig_2 = sig_1
    
    [Dpg,Ntot,limit,LIMIT] = Iterate_Dpg_FIX_2M(Dpg,sig,Ntot,N,limit,LIMIT,Dp,distrib, Dpg_1,Dpg_2)
    [sig,Ntot,limit,LIMIT] = Iterate_sig_FIX_2M(Dpg,sig,Ntot,N,limit,LIMIT,Dp,distrib, sig_1,sig_2)
    
    return [Dpg,sig,Ntot,limit,LIMIT]
    
def Iterate_Dpg_FIX_2M(Dpg_,sig_,Ntot_,N,limit_,LIMIT_,Dp,distrib, Dpg_1,Dpg_2):
    limit = limit_
    LIMIT = LIMIT_
    
    A = np.zeros((2,2))*np.nan
    F = np.zeros((2,1))*np.nan
    
    for i1 in range(0,N):
        A1 = f_lognorm(Dpg_1[i1], sig_[0], Dp)
        A[0,0] = np.sum(A1*A1)
        F[0,0] = np.sum(distrib*A1)
        
        for i2 in range(0,N):
            A2 = f_lognorm(Dpg_2[i2], sig_[1], Dp)
            A[1,1] = np.sum(A2*A2)
            A[1,0] = np.sum(A2*A1)
            A[0,1] = A[1,0]
            F[1,0] = np.sum(distrib*A2)
                
            Ntot__ = (np.linalg.inv(A).dot(F))

            if ((Ntot__[0]>=0) & (Ntot__[1]>=0)):
                fitting  = (Ntot__[0] * A1 + Ntot__[1] * A2) - distrib
                variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

                FITTING  = np.log10((Ntot__[0] * A1 + Ntot__[1]) / distrib)
                VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

                if ((variance <= limit)):
                    limit = variance
                    LIMIT = VARIANCE
                    Dpg = [Dpg_1[i1],Dpg_2[i2]]
                    Ntot = Ntot__
                    sig  = sig_
             
    if ((limit == limit_) or (LIMIT == LIMIT_)):
        Dpg = Dpg_
        Ntot = Ntot_
                               
    return [Dpg,Ntot,limit,LIMIT]

def Iterate_sig_FIX_2M(Dpg_,sig_,Ntot_,N,limit_,LIMIT_, Dp,distrib, sig_1,sig_2):
    A = np.zeros((2,2))*np.nan
    F = np.zeros((2,1))*np.nan
    
    limit = limit_
    LIMIT = LIMIT_
    
    for i1 in range(0,N):
        A1 = f_lognorm(Dpg_[0], sig_1[i1], Dp)
        A[0,0] = np.sum(A1*A1)
        F[0,0] = np.sum(distrib*A1)
        
        for i2 in range(0,N):
            A2 = f_lognorm(Dpg_[1], sig_2[i2], Dp)
            A[1,1] = np.sum(A2*A2)
            A[1,0] = np.sum(A2*A1)
            A[0,1] = A[1,0]
            F[1,0] = np.sum(distrib*A2)

            Ntot__ = (np.linalg.inv(A).dot(F))
            if ((Ntot__[0]>=0) & (Ntot__[1]>=0)):
                fitting  = (Ntot__[0] * A1 + Ntot__[1] * A2) - distrib
                variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

                FITTING  = np.log10((Ntot__[0] * A1 + Ntot__[1] * A2) / distrib)
                VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

                if ((variance <= limit)):
                    limit = variance
                    LIMIT = VARIANCE
                    sig = [sig_1[i1],sig_2[i2]]
                    Ntot = Ntot__
                        
    if ((limit == limit_) or (LIMIT == LIMIT_)):
        sig = sig_
        Ntot = Ntot_
    
    return [sig,Ntot,limit,LIMIT]


####################################### 1 MODE ###########################
def DO_FIT_400_1M(Dp, distrib):
    # Dp is the particle sizes from the first row of the mtrx / it has to be in meters !!
    
    ## STEP 01 of the Fitting:   Searching for proper parameters by using 'DO_FIT_scan.m'
    # defining initial fitting parameters, by automatic guessing according
    # to several physical concepts and assumptions:
    N = 50                     # Number of iterations
    limit = np.inf                 # starting least squares value
    LIMIT = np.inf                 # starting least squares value
    sig = np.array([1.3]) # Starting standared deviations
    
    # Following is constrained particle size ranges of each mode...
    Dp_min = np.array([3])*10**(-9)
    Dp_max = np.array([400])*10**(-9)

    Dpg,sig,Ntot,limit,LIMIT,dlogDpg = Dpg_scan_1M(distrib, Dp, N, limit, LIMIT, sig, Dp_min, Dp_max)
    sig,Ntot,limit,LIMIT = sig_scan_1M(Dpg,Ntot,N,limit,LIMIT,Dp,distrib, sig)
    N = 60
    for jj in range(0,2): # double scan ?
        Dp_min = 10**(np.log10(Dpg) - np.array([3,6]) * dlogDpg)
        Dp_max = 10**(np.log10(Dpg) + np.array([6,3]) * dlogDpg)

        Dpg,sig,Ntot,limit,LIMIT = DO_Fit_FIX_1M(distrib,Dp,N,limit,LIMIT,Dpg,sig,Ntot,Dp_min,Dp_max)
        
    return [Dpg, sig, Ntot, limit, LIMIT]


def Dpg_scan_1M(distrib, Dp, N, limit_, LIMIT_, sig_, Dp_min, Dp_max):
    
    dlogDpg = np.zeros(1)*np.nan
    # mode 1
    Dpg_1 = np.logspace(np.log10(Dp_max[0]), np.log10(Dp_min[0]), N, base=10.0, endpoint=True)
    dlogDpg[0] = np.log10(Dpg_1[1])-np.log10(Dpg_1[0])
    
    limit = limit_
    LIMIT = LIMIT_
    
    A = np.zeros((1,1))*np.nan
    F = np.zeros((1,1))*np.nan
    
    for i1 in range(0,N):
        A1 = f_lognorm(Dpg_1[i1], sig_[0], Dp)
        A[0,0] = np.sum(A1*A1)
        F[0,0] = np.sum(distrib*A1)
        
        Ntot__ = (np.linalg.inv(A).dot(F))

        if (Ntot__[0]>=0):
            fitting  = (Ntot__[0] * A1) - distrib
            variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

            FITTING  = np.log10((Ntot__[0] * A1) / distrib)
            VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))
            
#             if ((variance <= limit) & (VARIANCE <= LIMIT)):
            if ((variance <= limit)):
                limit = variance
                LIMIT = VARIANCE
                Dpg = [Dpg_1[i1]]
                Ntot = Ntot__
                sig  = sig_
                
        
                               
#   #if didnt improve the least mean squares, and run a finescanning
    if ((limit == limit_) or (LIMIT == LIMIT_)):
        sig = sig_
                               
    return [Dpg,sig,Ntot,limit,LIMIT,dlogDpg]

def sig_scan_1M(Dpg_,Ntot_,N,limit_,LIMIT_, Dp,distrib, sig_):
    sig_min = 1.1
    sig_max = 2.1
    dsig    = (sig_max - sig_min) / N
    sig_1 = np.arange(sig_min, sig_max+dsig, dsig)
    sig = sig_
    limit = limit_
    LIMIT = LIMIT_
    
    A = np.zeros((1,1))*np.nan
    F = np.zeros((1,1))*np.nan
    
    for i1 in range(0,N):
        A1 = f_lognorm(Dpg_[0], sig_1[i1], Dp)
        A[0,0] = np.sum(A1*A1)
        F[0,0] = np.sum(distrib*A1)

        Ntot__ = (np.linalg.inv(A).dot(F))
        if (Ntot__[0]>=0):
            fitting  = (Ntot__[0] * A1) - distrib
            variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

            FITTING  = np.log10((Ntot__[0] * A1) / distrib)
            VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

            if ((variance <= limit)):
                limit = variance
                LIMIT = VARIANCE
                sig = [sig_1[i1]]
                Ntot = Ntot__   

    if ((limit == limit_) or (LIMIT == LIMIT_)):
        sig = sig_
        Ntot = Ntot_
    
    return [sig,Ntot,limit,LIMIT]

def DO_Fit_FIX_1M(distrib,Dp,N,limit,LIMIT,Dpg,sig,Ntot,Dp_min,Dp_max):
    Dpg_1 = np.logspace(np.log10(Dp_max[0]), np.log10(Dp_min[0]), N, base=10.0, endpoint=True)
    
    sig_min = 1.1
    sig_max = 2.1
    dsig    = (sig_max - sig_min) / N
    sig_1 = np.arange(sig_min, sig_max+dsig, dsig)
    
    [Dpg,Ntot,limit,LIMIT] = Iterate_Dpg_FIX_1M(Dpg,sig,Ntot,N,limit,LIMIT,Dp,distrib, Dpg_1)
    [sig,Ntot,limit,LIMIT] = Iterate_sig_FIX_1M(Dpg,sig,Ntot,N,limit,LIMIT,Dp,distrib, sig_1)
    
    return [Dpg,sig,Ntot,limit,LIMIT]
    
def Iterate_Dpg_FIX_1M(Dpg_,sig_,Ntot_,N,limit_,LIMIT_,Dp,distrib, Dpg_1):
    limit = limit_
    LIMIT = LIMIT_
    
    A = np.zeros((1,1))*np.nan
    F = np.zeros((1,1))*np.nan
    
    for i1 in range(0,N):
        A1 = f_lognorm(Dpg_1[i1], sig_[0], Dp)
        A[0,0] = np.sum(A1*A1)
        F[0,0] = np.sum(distrib*A1)
        
        Ntot__ = (np.linalg.inv(A).dot(F))

        if (Ntot__[0]>=0):
            
            fitting  = (Ntot__[0] * A1) - distrib
            variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

            FITTING  = np.log10((Ntot__[0] * A1) / distrib)
            VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

            if ((variance <= limit)):
                limit = variance
                LIMIT = VARIANCE
                Dpg = [Dpg_1[i1]]
                Ntot = Ntot__
                sig  = sig_
             
    if ((limit == limit_) or (LIMIT == LIMIT_)):
        Dpg = Dpg_
        Ntot = Ntot_
                               
    return [Dpg,Ntot,limit,LIMIT]

def Iterate_sig_FIX_1M(Dpg_,sig_,Ntot_,N,limit_,LIMIT_, Dp,distrib, sig_1):
    A = np.zeros((1,1))*np.nan
    F = np.zeros((1,1))*np.nan
    
    limit = limit_
    LIMIT = LIMIT_
    
    for i1 in range(0,N):
        A1 = f_lognorm(Dpg_[0], sig_1[i1], Dp)
        A[0,0] = np.sum(A1*A1)
        F[0,0] = np.sum(distrib*A1)

        Ntot__ = (np.linalg.inv(A).dot(F))
        if (Ntot__[0]>=0):
            fitting  = (Ntot__[0] * A1) - distrib
            variance = np.sqrt(np.sum(fitting**2)/np.size(Dp))

            FITTING  = np.log10((Ntot__[0] * A1) / distrib)
            VARIANCE = np.sqrt(np.sum(FITTING**2)/np.size(Dp))

            if ((variance <= limit)):
                limit = variance
                LIMIT = VARIANCE
                sig = [sig_1[i1]]
                Ntot = Ntot__
                        
    if ((limit == limit_) or (LIMIT == LIMIT_)):
        sig = sig_
        Ntot = Ntot_
    
    return [sig,Ntot,limit,LIMIT]

###### functions to plot and test PNSD fitting ######

import matplotlib.pyplot as plt

def thickax(ax):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.3)
        ax.spines[axis].set_color('k')
    plt.rc('axes', linewidth=0.2)
    fontsize = 12
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    ax.tick_params(direction='out', length=8, width=1.3, pad=10, bottom=True, top=False, left=True, right=False, color='k')
    ax.tick_params(which='minor', length=4, color='k', width=1.3)
    
def f_lognorm2(Dpg,sig,Ntot,Dp):
    return Ntot * 0.39894228 / np.log10(sig) * np.exp(-0.5*(np.log10(Dp)-np.log10(Dpg))**2/(np.log10(sig)**2))

def model_distrib_3M(Dp, distrib):
    Dpg, sig, Ntot = DO_FIT_400_3M(Dp, distrib, finescanning='no')[0:3]
    return [f_lognorm2(Dpg[0],sig[0],Ntot[0],Dp), f_lognorm2(Dpg[1],sig[1],Ntot[1],Dp), 
            f_lognorm2(Dpg[2],sig[2],Ntot[2],Dp)]

def model_distrib(Dp, distrib):
    Dpg, sig, Ntot = MAIN_FIT(Dp, distrib, finescanning='no')[0:3]
    return [f_lognorm2(Dpg[i],sig[i],Ntot[i],Dp) for i in range(len(Dpg))]

def model_param(Dp, distrib):
    Dpg, sig, Ntot = DO_FIT_400_3M(Dp, distrib, finescanning='no')[0:3]
    return [Dpg,sig,Ntot]

def plot_fit(Dp_, distrib, estimate, ax):
    for i in range(len(estimate)):
        plt.plot(Dp_,estimate[i], '--', label='fit mode ' + str(i), color='b')
    plt.plot(Dp_,sum(estimate), label='fit', color='b', linewidth=3, alpha=0.6)
    plt.plot(Dp_,distrib, color='r', label='measured', linewidth=3, alpha=0.6)
    ax.set_xscale('log')
    plt.legend()
    ax.grid(True, alpha=0.6)
    ax.set_xlabel(r'Dp ($nm$)', fontsize=15)
    thickax(ax)
    ax.set_ylabel(r'dN/dlogDg ($cm^{-3}$)', fontsize=15)
    ax.set_xlim([10,410])
    ax.set_xticks([10,20,30,40,50,60,70,80,90,100,200,300,400])
    ax.set_xticklabels([10,20,30,'','','','','','',100,'','',400])
