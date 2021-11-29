########################################
## QCD constants and some definitions ##
########################################

import numpy as np
import scipy as sp
import math
import sys
import scipy.special as special
from random import seed
from random import gauss
from random import random

print('The length of the medium, in fm, is:', sys.argv[1]) 
print('The jet radius is:', sys.argv[2]) 
print('The jet pt, in GeV, is:', sys.argv[3]) 
class Definitions:

    def __init__(self):
    # Jet and DyG definition
        self.R_jet = float(sys.argv[2])
        self.pt_jet = float(sys.argv[3])
        self.Q = self.pt_jet*self.R_jet

    # QCD parameters
        self.Cf = 4./3.
        self.Ca = 3.
        self.tr = 0.5
        self.nf = 5.
        self.b0 = (11.*self.Ca-4.*self.nf*self.tr)/(12.*math.pi)
        self.alphas_mz = 0.1265
        self.mz = 91.1876
        self.mu_freezing = 1. #GeV
        self.alphas_vac = 0.2 #alphas for VLEs

    # Medium-related parameters
        self.alphas_med = 0.24 # alphas for MIEs
        self.GeVtoFm = 0.197
        self.wBH = 0.45 # temperature of the medium in GeV
        self.qhat0 = 1.5*self.GeVtoFm # GeV^3 medium qhat
        self.xi = float(sys.argv[1])/self.GeVtoFm # path length  
        self.omega_c = 0.5*self.qhat0*pow(self.xi,2)
        self.theta_c = 2./np.sqrt(self.qhat0*pow(self.xi,3))
        self.Qs = np.sqrt(self.qhat0*self.xi); 

    # Other functions that we will need
    def step(self,x):
        return 1 * (x >= 0)

    def incomplete_gamma(self,x): # For a = 0-> Gamma(0,x) = \int_x^infty  dt e^-t/t See https://fr.wikipedia.org/wiki/Fonction_gamma_incomplete
        return special.exp1(x)
    
    def veto_region(self,z,theta):
        return self.step(theta*self.R_jet-self.theta_c)*self.step(z*self.pt_jet*pow(self.R_jet*theta,2)*self.xi-2.)*self.step(2*self.qhat-pow(z*self.pt_jet,3)*pow(theta*self.R_jet,4))

