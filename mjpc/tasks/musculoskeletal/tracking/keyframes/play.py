# load E:\github_code\mujoco_mpc\mjpc\tasks\musculoskeletal\Models\mj_FullBody\mj_fullbody_simple.xml in mujoco
import mujoco
import mujoco.viewer
import os
import sys
import numpy as np
import time
os.chdir('/home/zsn/research/mpc/mujoco_mpc_yyw/mjpc/tasks/musculoskeletal/Models/mj_FullBody')
model = mujoco.MjModel.from_xml_path('mj_fullbody_simple.xml')
# os.chdir(r'E:\github_code\FullBody_MS_Model\mj_full')
# model = mujoco.MjModel.from_xml_path('mj_fullbody_walk.xml')
data = mujoco.MjData(model)

import xml.etree.ElementTree as ET

def read_joint_trajectory(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    trajectory = {}
    
    for key in root.findall('.//key'):
        name = key.get('name')
        time = float(key.get('time'))
        qpos = [float(x) for x in key.get('qpos').split()]
        qvel = [float(x) for x in key.get('qvel').split()]
        
        trajectory[name] = {
            'time': time,
            'qpos': qpos,
            'qvel': qvel
        }
    
    return trajectory

# Assuming the XML file is in the same directory as the script
xml_file_path = '/home/zsn/research/mpc/mujoco_mpc_yyw/mjpc/tasks/musculoskeletal/tracking/keyframes/whole_joint_trajectory_velocity.xml'
joint_trajectory = read_joint_trajectory(xml_file_path)

# Create a MuJoCo viewer
viewer = mujoco.viewer.launch_passive(model, data)

# Main simulation loop
i = 0
data.qpos[:] = joint_trajectory['walk_0']['qpos']

while viewer.is_running() and i < len(joint_trajectory):
    frame_name = f"walk_{i}"
    if frame_name in joint_trajectory:
        frame = joint_trajectory[frame_name]
        
        # Set joint positions
        # data.qpos[:] = frame['qpos']
        
        # Set joint velocities
        data.qvel[:] = frame['qvel']
        
        data.qacc[:] = np.zeros_like(data.qacc)
    else:
        print(f"Frame {frame_name} not found in trajectory")
        break

    # Forward the simulation
    # mujoco.mj_fwdPosition(model, data)
    mujoco.mj_step(model, data)
    # Update the viewer
    viewer.sync()

    # Sleep to control playback speed
    time.sleep(0.002)  # Adjust this value to change playback speed

    i += 1

# Close the viewer
viewer.close()
