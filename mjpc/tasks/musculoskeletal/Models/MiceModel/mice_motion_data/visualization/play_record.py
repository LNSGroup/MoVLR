import mujoco
import os
import numpy as np
import time
import imageio

# ... existing imports ...

os.chdir(os.path.dirname(os.path.abspath(__file__)))

### Initialize model and Read joint trajectory
mujoco_model_path = '../../CyberMice_Marker.xml'
model = mujoco.MjModel.from_xml_path(mujoco_model_path)
data = mujoco.MjData(model)
qpos_file_path = '../ik_results/logs/combined_qpos.npy'
qpos_array = np.load(qpos_file_path)
# act_file_path = 'logs/txt/act_dynsyn_walk.npy'
# act_array = np.load(act_file_path)

print("qpos_array.shape: ", qpos_array.shape)
# print("act_array.shape: ", act_array.shape)

### Main simulation loop
i = 0
data.qpos = qpos_array[0]
mujoco.mj_fwdPosition(model, data)

frames_list = []

rgb_renderer = mujoco.Renderer(model, width=1302, height=1080)
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True
time_step = 1

while i < qpos_array.shape[0]:
    # Set joint positions and activations
    data.qpos = qpos_array[i]
    # data.act = act_array[i]
    data.qvel = np.zeros_like(data.qvel)
    mujoco.mj_fwdPosition(model, data)

    # Render and save frame
    rgb_renderer.update_scene(data, camera='lateral_camera', scene_option=scene_option)
    mujoco_frame = rgb_renderer.render()
    frames_list.append(mujoco_frame)

    i += time_step
    print("Frame:", i)

time_str = time.strftime('%m%d_%H_%M_%S')
imageio.mimsave(f'../ik_results/logs/video_{time_str}.mp4', frames_list, fps=50)
print("Successfully saved video")
rgb_renderer.close()
