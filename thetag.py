###################################
###   The theta-g distribution  ###
###################################

import numpy as np
import scipy as sp
import math
import sys
import scipy.special as special
from scipy.integrate import quad
from basics import Definitions
from elements import *

spectrum = Spectrum() # Call the class to use one of its functions
eloss = Eloss()

# Class for the theta_g distribution without energy loss
class ThetaDyG_noEloss(Sudakov): 
    def __init__(self):
        super().__init__()
    
    def vacuum_int(self,z,theta,a,fl='q'):
        kappa = z*pow(theta/self.R_jet,a)
        return self.Pvac(z,theta,fl)*self.DeltaVac(kappa,a,fl)

# vacuum splittings only
    def vacuum(self,theta,a,fl='q'): 
        zmin = 0.
        zmax = 1.0
        return quad(self.vacuum_int,zmin,zmax,args=(theta,a,fl))[0]

# VLEs only
    def vacuum_int_veto(self,z,theta,a,fl='q'):
        kappa = z*pow(theta/self.R_jet,a)
        return self.Pvac(z,theta,fl)*self.DeltaVac_veto(kappa,a,fl)

    def vacuum_veto(self,theta,a,fl='q'): 
        #Area_full: z*(theta/R)^a > kappa
        res_full = quad(self.vacuum_int_veto,0,1,args=(theta,a,fl))[0]
        #Area veto: td < tf < L
        thetamin = max(np.sqrt(2./(self.pt_jet*self.xi)), self.theta_c)
        zmin = 2./(theta**2*self.pt_jet*self.xi)
        zmax = min(1, (2.*self.qhat0/(self.pt_jet**3*theta**4))**(1./3.))
        res_veto = 0
        if thetamin<=theta<=self.R_jet and zmin<zmax: 
            res_veto = quad(self.vacuum_int_veto,zmin,zmax,args=(theta,a,fl),epsrel=1e-8)[0]
        return res_full - res_veto

# MIEs only
    def mie_int(self,z,theta,a,fl='q'):
        kappa = z*pow(theta/self.R_jet,a)
        return self.Pmed(z,fl)*self.Pbroad(z,theta)*self.DeltaMed(kappa,a,fl)

    def mie(self,theta,a,fl='q'): 
        zmin = 0.
        zmax = np.minimum(self.omega_c/self.pt_jet,1.0);
        return quad(self.mie_int,zmin,zmax,args=(theta,a,fl),epsrel=1e-8)[0]/(1.-self.DeltaMed(1e-8,a,fl))

# Toy model  
    def medium_int_veto(self,z,theta,a,fl='q'): 
        kappa = z*pow(theta/self.R_jet,a)
        #Area veto: td < tf < L
        thetamin = max(np.sqrt(2./(self.pt_jet*self.xi)), self.theta_c)
        zmin = 2./(theta**2*self.pt_jet*self.xi)
        zmax = min(1, (2.*self.qhat0/(self.pt_jet**3*theta**4))**(1./3.))
        if thetamin<=theta<=self.R_jet and zmin<=z<=zmax: 
            pvac = 0.0
        else:
            pvac = self.Pvac(z,theta,fl)
        return (pvac+self.Pmed(z,fl)*self.Pbroad(z,theta))*self.DeltaMed(kappa,a,fl)*self.DeltaVac_veto(kappa,a,fl)

    def medium_veto(self,theta,a, fl='q'):
        res_full = quad(self.medium_int_veto,0,1,args=(theta,a,fl),epsrel=1e-4)[0]
        return res_full 

# Theta-g distribution including realistic q/g fractions  
    def thetag_matched(self,theta,a):
        quark_xs = spectrum.fit(self.pt_jet,'q','Pb')[0]
        gluon_xs = spectrum.fit(self.pt_jet,'g','Pb')[0]
        total_xs = quark_xs + gluon_xs
        return (quark_xs*self.vacuum(theta,a,'q')+self.vacuum(theta,a,'g')*gluon_xs)/total_xs

    def thetag_matched_med(self,theta,a):
        quark_xs = spectrum.fit(self.pt_jet,'q','p')[0]
        gluon_xs = spectrum.fit(self.pt_jet,'g','p')[0]
        total_xs = quark_xs + gluon_xs
        return (quark_xs*self.medium_veto(theta,a,'q')+self.medium_veto(theta,a,'g')*gluon_xs)/total_xs

# Class for the theta_g distribution with energy loss
class ThetaDyG_wEloss(ThetaDyG_noEloss): 

    def __init__(self):
        super().__init__()

    def thetag_matched_qw_eloss(self,theta,a): #beware that this is not normalised
        eloss_q = eloss.qw_eloss(theta,'q')
        eloss_g = eloss.qw_eloss(theta,'g')
        quark_xs = spectrum.fit(self.pt_jet,'q','Pb')[0]
        gluon_xs = spectrum.fit(self.pt_jet,'g','Pb')[0]
        return eloss_q*quark_xs*self.medium_veto(theta,a,'q')+eloss_g*gluon_xs*self.medium_veto(theta,a,'g')

    def normalisation(self,a):
        thetamin = 0.;
        thetamax = self.R_jet;
        return quad(lambda theta: self.thetag_matched_qw_eloss(theta,a),thetamin,thetamax,epsrel=1e-4)[0]
        