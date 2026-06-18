import numpy as np
import pickle
import torch
from plot_3d_global import scaling
from pos_ik import qpos_ik

folder = './motion_traj'
file_name = 'swim_an_hour'
data = np.load(f'{folder}/{file_name}.npy')

print(data.shape)

print(data[0, 0])
# raise NotImplementedError
data = scaling(data)

print(data.shape)

qpos_traj = qpos_ik(data[0])
np.save(f'{file_name}_qpos.npy', qpos_traj)
# open /home/yunyue/Desktop/Git_Repo/mpc2/mjpc/tasks/musculoskeletal/Models/MS_Human_700_Release/qpos_traj.pkl
# with open('/home/yunyue/Desktop/Git_Repo/mpc2/mjpc/tasks/musculoskeletal/Models/MS_Human_700_Release/qpos_traj.pkl', 'r') as f:
#     qpos_traj = pickle.load(f)

# print(qpos_traj.shape)

# qpos_data = np.load('qpos_traj.npy')    
# print(qpos_data.shape)