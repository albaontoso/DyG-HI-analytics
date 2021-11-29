import numpy as np
import scipy as sp
import math
import sys
import scipy.special as special
from scipy.integrate import quad
from basics import Definitions
from elements import *
from thetag import *

thetag_ = ThetaDyG_noEloss()
definitions = Definitions()
# Define a value of the Dynamical grooming parameter 'a' and the angle
a = 1
thetag_value = 0.2
print('Results for: 1/\sigma d\sigma/dtheta_g')
print('---------------------------------------') 
print('Vacuum:', thetag_.vacuum(thetag_value,a))
print('VLEs:', thetag_.vacuum_veto(thetag_value,a))
print('MIEs:', thetag_.mie(thetag_value,a))
print('Toy model:', thetag_.medium_veto(thetag_value,a))