########################################
###   Vacuum/In-medium DyG elements  ###
########################################

import numpy as np
import scipy as sp
import math
import sys
import scipy.special as special
from scipy.integrate import quad
from basics import Definitions

# Class for the branching kernels
class BranchKernel(Definitions):
    def __init__(self):
        super().__init__()

# Splitting function at DLA
    def Pvac(self,z,theta,fl='q'):
        if fl=='q':
            Ci = self.Cf
        if fl=='g':
            Ci = self.Ca
        return 2.0*self.alphas_vac*Ci/(theta*z*math.pi)

# Splitting function a la BDMPS
    def Pmed(self,z,fl='q'):
        if fl=='q':
            Ci=self.Cf
        if fl=='g':
            Ci=self.Ca

        alpha_bar = self.alphas_med*Ci/math.pi
        return alpha_bar*np.sqrt(2*self.omega_c/self.pt_jet)*pow(z,-3./2.)


# Broadening function a la BDMPS
    def Pbroad(self,z,theta):
        return 2*theta*pow(z*self.pt_jet/self.Qs,2)*self.incomplete_gamma(pow(z*self.pt_jet*theta/self.Qs,2)) 

# Class for the Sudakov form factors
class Sudakov(BranchKernel):
    def __init__(self):
        super().__init__()

# Sudakov in the vacuum
    def DeltaVac(self,kappa,a,fl='q'):
        if fl=='q':  
            Ci = self.Cf
        if fl=='g':
            Ci = self.Ca;

        alpha_bar = self.alphas_vac*Ci/math.pi 
         #Area full: z*(theta/R)^a > kappa
        Afull = alpha_bar*pow(np.log(kappa),2)/a
        return np.exp(-Afull) 

# Sudakov for VLEs          
    def DeltaVac_int_veto(self,theta,kappa,a,fl='q'):
        if fl=='q':  
            Ci = self.Cf
        if fl=='g':
            Ci = self.Ca;
        zmin = max(kappa*pow(self.R_jet/theta,a), 2./(pow(theta,2)*self.pt_jet*self.xi))    
        zmax = min(1.0,pow(2.*self.qhat0/(pow(self.pt_jet,3)*pow(theta,4)),1./3.))
        alpha_bar = self.alphas_vac*Ci/math.pi 
        if zmax>zmin:
            return 2*alpha_bar*np.log(zmax/zmin)/theta
        else:
            return 0.0

    def DeltaVac_veto(self,kappa,a,fl='q'):
        if fl=='q':  
            Ci = self.Cf
        if fl=='g':
            Ci = self.Ca;
        alpha_bar = self.alphas_vac*Ci/math.pi 
         #Area full: z*(theta/R)^a > kappa
        Afull = alpha_bar*pow(np.log(kappa),2)/a
        #Area veto: td < tf < L
        thetamin = max(self.R_jet*pow(kappa,1./a),np.sqrt(2./(self.pt_jet*self.xi)),self.theta_c,pow(2.*self.qhat0/pow(kappa*pow(self.R_jet,a)*self.pt_jet,3),1./(4.-3.*a)) if a>4./3. else 0)
        thetamax = min(self.R_jet,pow(2.*self.qhat0/pow(kappa*pow(self.R_jet,a)*self.pt_jet,3),1./(4.-3.*a)) if a<=4./3. else self.R_jet)
        Aveto = 0
        if thetamin < thetamax:
            Aveto = quad(self.DeltaVac_int_veto,thetamin,thetamax,args=(kappa,a,fl))[0]
        return np.exp(-(Afull-Aveto))

# Sudakov for MIEs 
    def DeltaMed_int(self,z,kappa,a,fl='q'):
        if fl=='q':  
            Ci = self.Cf
        if fl=='g':
            Ci = self.Ca;
        Qs2 = pow(self.Qs,2)
        X2 = pow(z*self.pt_jet*self.R_jet,2)/Qs2
        tmin2 = pow(kappa/z,2./a)
        return self.Pmed(z,fl)*(X2*self.incomplete_gamma(X2)-np.exp(-X2)+np.exp(-tmin2*X2)-X2*tmin2*self.incomplete_gamma(X2*tmin2))
        
    def DeltaMed(self,kappa,a,fl='q'): 
        if kappa > self.omega_c/self.pt_jet:
            return 1
        else:
            zmin = kappa
            zmax = np.minimum(self.omega_c/self.pt_jet,1.0)
            arg = quad(self.DeltaMed_int,zmin,zmax,args=(kappa,a,fl))[0]  
            return np.exp(-arg)  

# Class for the jet spectrum parametrization
class Spectrum(Definitions): 

    def __init__(self):
       super().__init__()

    def fit(self,pt,fl='q',A='p'): # We return the spectrum and the exponent itself for different flavors and nuclear species
        if A=='p': # proton
            if fl=='q':
                a = 0.008332946739830507; 
                pt0 = 20.828393199777093;
                b = 4.7335957582977946;
                c = 0.07710153418923364;
                d = -0.05059119544424069;
                e = 0.030686480100498965;

            if fl=='g':
                a = 0.007238048478317841; 
                pt0 = 26.40241480367206;
                b = 5.137396353863606;
                c = 0.1652735112639996;
                d = -0.08016012722958223;
                e = 0.04538279533856495;

        if A=='Pb': # proton
            if fl=='q':
                a = 0.0009540659548422345; 
                pt0 = 32.82803487818042;
                b = 4.681529743242485;
                c = 0.09808186013520422;
                d = -0.002055950046580344;
                e = 0.03245148331457684;

            if fl=='g':
                a = 0.0005184234772818768; 
                pt0 = 44.42440252697745;
                b = 5.229161329428604;
                c = 0.15483157310048482;
                d = -0.006811936662039507;
                e = 0.05351278619070584;

        return [a*pow(pt0/pt, b+c*np.log(pt/pt0)+d*np.log(pt/pt0)**2+e*np.log(pt/pt0)**3),b+c*np.log(pt/pt0)+d*np.log(pt/pt0)**2+e*np.log(pt/pt0)**3]

# Class to define energy loss models
class Eloss(Spectrum): # Toy E-loss model based on the quenching weights formalism
    def __init__(self):
        super().__init__()
     
    # Quenching weights
    def Q0(self, fl='q'):
        if fl=='q':  
            Ci = self.Cf
        if fl=='g':
            Ci = self.Ca
        omega_max = min(self.Qs/self.R_jet,self.omega_c)
        n = self.fit(self.pt_jet,fl,'Pb')[1]
        nu = n/self.pt_jet;
        prefactor = 2.*self.alphas_med*Ci*np.sqrt(2.*self.omega_c/omega_max)/math.pi
        arg = 1.0-np.sqrt(math.pi*nu*omega_max)*special.erf(np.sqrt(nu*omega_max))-np.exp(-nu*omega_max)
        return np.exp(prefactor*arg)
    
    def qw_eloss(self,theta,fl='q'):
        return self.Q0(fl)*self.step(self.theta_c-theta)+self.step(theta-self.theta_c)*self.Q0(fl)*self.Q0('g')


