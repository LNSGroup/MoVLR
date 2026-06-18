# from ik import *
from dm_control.utils.inverse_kinematics import *
import numpy as np
import pickle
from dm_control import mujoco
import os

# traj = loaded_arr = np.load('../0_out.npy')[0]

def qpos_ik(traj):

    MINS = traj.min(axis=0).min(axis=0)
    MAXS = traj.max(axis=0).max(axis=0)


    height_offset = MINS[1]
    print("height offset", height_offset)
    # print("mins", MINS)
    traj[:, :, 1] -= height_offset
    trajec = traj[:, 0, [0, 2]]

    traj[..., 0] -= traj[:, 0:1, 0]
    traj[..., 2] -= traj[:, 0:1, 2]
    print(trajec)

    traj[:,:,[0,1,2]] = traj[:,:,[2,0,1]]

    qpos_traj = []

    for i in range(traj.shape[0]):

        frame = traj[i]
        # print(frame, frame.shape)

        unused_indices = [3, 6, 9, 15]
        frame = np.delete(frame, unused_indices, axis=0)
        # print("new frame", frame, frame.shape)

        target_pos = frame
        # print(target_pos, target_pos.shape)

        model_path = "/home/yunyue/Desktop/Git_Repo/mpc2/mjpc/tasks/musculoskeletal/Models/MS_Human_700_Release/mj_fullbody_water.xml"
        xml_string = open(model_path, 'r').read()

        physics = mujoco.Physics.from_xml_string(xml_string)

        sites_string = ['pelvis', 'femur_l', 'femur_r', 'tibia_l', 'tibia_r', 'talus_l', 'talus_r', 'toes_l', 'toes_r', 'head_neck', 'clavicle_l', 'clavicle_r', 'scapula_l', 'scapula_r', 'ulna_l', 'ulna_r', 'proximal_row_l', 'proximal_row_r']

        
        result = qpos_from_site_pose(
            physics=physics,
            sites_names=sites_string,
            target_pos=target_pos,
            joint_names=None,
            tol=1e-14,
            regularization_threshold=0.5,
            regularization_strength=1e-2,
            max_update_norm=2.0,
            progress_thresh=5000.0,
            max_steps=1000,
            inplace=False,
            null_space_method=False
        )
        
        qpos_traj.append(result.qpos)
    # print(qpos_traj, len(qpos_traj))
    qpos_traj = np.array(qpos_traj)
    qpos_traj [:, [1,2]] += trajec[:, [0,1]]
    print(qpos_traj[30:40])
    with open('qpos_traj.pkl', 'wb') as file:
        pickle.dump(qpos_traj, file)