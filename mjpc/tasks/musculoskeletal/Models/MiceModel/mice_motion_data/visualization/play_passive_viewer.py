import mujoco
import os
import mujoco.viewer
import numpy as np
import time
import imageio

# ... existing imports ...

os.chdir(os.path.dirname(os.path.abspath(__file__)))

### Initialize model and Read joint trajectory
mujoco_model_path = '../../CyberMice_Marker.xml'
model = mujoco.MjModel.from_xml_path(mujoco_model_path)
data = mujoco.MjData(model)
qpos_file_path = '../ik_results/joint_qpos.npy'
qpos_array = np.load(qpos_file_path)
# act_file_path = 'logs/txt/act_dynsyn_walk.npy'
# act_array = np.load(act_file_path)

print("qpos_array.shape: ", qpos_array.shape)
# print("act_array.shape: ", act_array.shape)

# ... existing code ...

### Main simulation loop
i = 0
data.qpos = qpos_array[0]
mujoco.mj_fwdPosition(model, data)

# Create MuJoCo viewer
viewer = mujoco.viewer.launch_passive(model, data)

time_step = 1

while i < qpos_array.shape[0]:
    # Set joint positions and activations
    data.qpos = qpos_array[i]
    # data.act = act_array[i]
    data.qvel = np.zeros_like(data.qvel)
    mujoco.mj_fwdPosition(model, data)

    # Update the viewer
    viewer.sync()

    # Add a small delay to control playback speed
    time.sleep(0.0)

    i += time_step
    print("Frame:", i)
viewer.close()
