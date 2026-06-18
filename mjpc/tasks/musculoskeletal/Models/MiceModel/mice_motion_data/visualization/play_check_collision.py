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
qpos_file_path = '../ik_results/logs/joint_qpos.npy'
qpos_array = np.load(qpos_file_path)
# act_file_path = 'logs/txt/act_dynsyn_walk.npy'
# act_array = np.load(act_file_path)
feet_body_name_list = ['LPedal','RPedal','LCarpi','RCarpi']
feet_body_id_list = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in feet_body_name_list]
feet_geom_id_list = []
for body_name in feet_body_name_list:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == body_id:
            feet_geom_id_list.append(geom_id)
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
    data.qpos[2] -=0.002
    data.qvel = np.zeros_like(data.qvel)
    mujoco.mj_fwdPosition(model, data)

    # Update the viewer
    viewer.sync()
    
    # Add a small delay to control playback speed
    # time.sleep(0.01)

    # Check for collisions between feet and ground
    mujoco.mj_forward(model, data)
    for body_id in feet_body_id_list:
        for contact in data.contact:
            if (contact.geom1 == 0 and contact.geom2 in range(model.body_geomadr[body_id],model.body_geomadr[body_id]+model.body_geomnum[body_id])) or \
                (contact.geom2 == 0 and contact.geom1 in range(model.body_geomadr[body_id],model.body_geomadr[body_id]+model.body_geomnum[body_id])):
                print(f"Collision detected: {feet_body_name_list[feet_body_id_list.index(body_id)]} at frame {i}")
                break

    i += time_step
    print("Frame:", i)
viewer.close()
