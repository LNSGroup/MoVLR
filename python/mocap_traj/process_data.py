import numpy as np
from plot_3d_global import scaling
from pos_ik import qpos_ik
# from dm_control.utils.inverse_kinematics import *

# from 

data = np.load('./diving.npy')

print(data.shape)

data = scaling(data)

print(data.shape)

qpos_ik(data[0])