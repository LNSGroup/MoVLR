import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../mink')))

from pathlib import Path
import numpy as np
import h5py
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / ".." / "CyberMice_Marker.xml"

marker_list = ['Nose', 'LEar', 'REar', 'C1', 'L1', 'Pelvis', 'CA4',\
                'LScapula', 'LHumerus', 'LUlna', 'LWrist',\
                'RScapula', 'RHumerus', 'RUlna', 'RWrist',\
                'LFemur', 'LLeg', 'LPedal_Wrist', 'LPedal',\
                'RFemur', 'RLeg', 'RPedal_Wrist', 'RPedal']

marker_weight = {
    'Nose': 50.0, 'LEar': 50.0, 'REar': 50.0, 'C1': 20.0, 'L1': 20.0, 'Pelvis': 20.0, 'CA4': 20.0,\
    'LScapula': 20.0, 'LHumerus': 50.0, 'LUlna': 50.0, 'LWrist': 200.0,\
    'RScapula': 20.0, 'RHumerus': 50.0, 'RUlna': 50.0, 'RWrist': 200.0,\
    'LFemur': 50.0, 'LLeg': 50.0, 'LPedal_Wrist': 200.0, 'LPedal': 200.0,\
    'RFemur': 50.0, 'RLeg': 50.0, 'RPedal_Wrist': 200.0, 'RPedal': 200.0
}

class trajectory:
    def __init__(self, data_name='data/2020_12_22_1.h5'):
        
        file_path = os.path.dirname(__file__)
        data_path = os.path.join(file_path, data_name)
        file = h5py.File(data_path, 'r')
        group_pose = file['pose']
        self.dataset_keypoints = group_pose['keypoints']                 # (360000, 3, 23), 23 keypoints

        self.sample_rate = 50  # Hz

    def query(self, sim_time):
        frame_index = int(sim_time * self.sample_rate)
        data = self.dataset_keypoints[frame_index, :, :]
        data = data * 0.001

        data[0,:] = data[0,:] * 0.5
        data[1,:] = data[1,:] * 0.5
        data[2,:] = data[2,:] * 0.5
        return np.array(data)       # (3, 23)
    
class recorder:
    def __init__(self, marker_list, mj_model):
        self.marker_list = marker_list
        self.nmarker = len(marker_list)
        self.jnt_name_list = [mj_model.joint(jnt_id).name for jnt_id in range(mj_model.njnt)]
        self.njnt = mj_model.njnt

        self.marker_xpos_list = []
        self.joint_qpos_list = []
        self.time_list = []

    def record(self, mj_data, sim_time):
        marker_xpos = []
        for marker in self.marker_list:
            marker_xpos.append(mj_data.site('Marker_' + marker).xpos.copy())
        self.marker_xpos_list.append(marker_xpos)

        self.joint_qpos_list.append(mj_data.qpos.copy())

        self.time_list.append(sim_time)

    def output(self, output_dir=_HERE / "ik_results"):
        # to 2 .csv files
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        marker_xpos_array = np.array(self.marker_xpos_list)
        joint_qpos_array = np.array(self.joint_qpos_list)

        # output marker xpos with labels
        marker_xpos_file = os.path.join(output_dir, 'marker_xpos.csv')
        output_labels = [marker + '_x,' + marker + '_y,' + marker + '_z' for marker in self.marker_list]
        output_labels = ','.join(output_labels)
        # add time labels
        output_labels = 'time,' + output_labels
        time_array = np.array(self.time_list).reshape(-1, 1)
        marker_xpos_array = marker_xpos_array.reshape(-1, self.nmarker*3)
        output_array = np.concatenate((time_array, marker_xpos_array), axis=1)
        np.savetxt(marker_xpos_file, output_array, delimiter=',', header=output_labels, comments='')

        # output joint qpos with labels
        joint_qpos_file = os.path.join(output_dir, 'joint_qpos.csv')
        output_labels = [jnt_name for jnt_name in self.jnt_name_list]
        output_labels = ','.join(output_labels)
        output_labels = 'time,' + output_labels
        output_array = np.concatenate((time_array, joint_qpos_array), axis=1)
        np.savetxt(joint_qpos_file, output_array, delimiter=',', header=output_labels, comments='')

    

if __name__ == "__main__":
    marker_trajectory = trajectory()
    
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    ik_recorder = recorder(marker_list, model)

    configuration = mink.Configuration(model)

    task_list = []
    for index in range(len(marker_list)):
        marker_site = 'Marker_' + marker_list[index]
        weight = marker_weight[marker_list[index]]
        task = mink.FrameTask(frame_name=marker_site, frame_type="site", position_cost=weight, orientation_cost=0.0, lm_damping=1.0)
        task_list.append(task)

    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=True, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        rate = RateLimiter(frequency=200.0, warn=False)
        sim_time = 0.0
        start_sim_time = 20.0
        sim_time_step = 0.02
        sim_time_max = 20.0
        while viewer.is_running():
            # Update task targets
            print("Sim time: {:.2f}".format(sim_time))
            translation = marker_trajectory.query(sim_time=sim_time+start_sim_time)
            for index in range(len(task_list)):
                data.mocap_pos[index] = translation[:, index]
                task_translation = translation[:, index]
                se3_target = mink.SE3.from_rotation_and_translation(rotation=mink.SO3.identity(), translation=task_translation)
                task_list[index].set_target(se3_target)

            vel = mink.solve_ik(configuration, task_list, rate.dt, solver, 1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            ik_recorder.record(data, sim_time)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
            sim_time += sim_time_step                

            if sim_time > sim_time_max:
                ik_recorder.output()
                break
