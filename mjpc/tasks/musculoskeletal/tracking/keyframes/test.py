
import xml.etree.ElementTree as ET
import numpy as np
import os
os.chdir('/home/zsn/research/mpc/mujoco_mpc_yyw/mjpc/tasks/musculoskeletal/tracking/keyframes')
# Parse the XML file
tree = ET.parse('trajectory_keyframes.xml')
root = tree.getroot()

# Find all key elements
keys = root.findall('.//key')

# Initialize lists to store data
times = []
positions = []
velocities = []
# Extract data from each key element
for key in keys:
    time = float(key.get('time'))
    qpos = [float(x) for x in key.get('qpos').split()]
    qvel = [float(x) for x in key.get('qvel').split()]
    times.append(time)
    positions.append(qpos)
    velocities.append(qvel)
    
# Convert lists to numpy arrays
times = np.array(times)
positions = np.array(positions)
velocities = np.array(velocities)
# Create a structured numpy array
trajectory = positions
trajectory_velocities = velocities

# load E:\github_code\mujoco_mpc\mjpc\tasks\musculoskeletal\Models\mj_FullBody\mj_fullbody_simple.xml in mujoco
import mujoco
import mujoco.viewer
import os
import sys
import time
os.chdir('/home/zsn/research/mpc/mujoco_mpc_yyw/mjpc/tasks/musculoskeletal/Models/mj_FullBody')
model = mujoco.MjModel.from_xml_path('mj_fullbody_simple.xml')
# os.chdir(r'E:\github_code\FullBody_MS_Model\mj_full')
# model = mujoco.MjModel.from_xml_path('mj_fullbody_walk.xml')
data = mujoco.MjData(model)

## DOF in muscle1000
dof_pelvis = [i for i in range(6)]
dof_leg_r = [6, 7, 8, 12, 15, 16, 17, 18]  # hip_flexion to subtalar_angle
dof_foot_r = [i for i in range(19, 41)]    # midtarsal_angle to itp_dist_5
dof_leg_l = [i + 38 for i in dof_leg_r]
dof_foot_l = [i + 38 for i in dof_foot_r]
dof_torso = [85, 86, 87, 100, 101, 102, 136, 137, 138, 154, 155, 156]  # spine dofs
dof_arm_r = [170, 171, 173, 174, 175, 176, 177]    # elv_angle to flexion
dof_arm_l = [i + 20 for i in dof_arm_r]

## DOF in muscle 700
dof_muscle_700 = np.array([0,1,2,3,4,5,6, 7, 8, 12, 15, 16, 17, 21, 22, 23, 27, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 55, 56, 58, 59, 60, 61, 62, 75, 76, 78, 79, 80, 81, 82])


# trajectory indexs in muscle 700
active_joint_indexs_700 = np.array([0,1,2,3,4,5]+[6, 7, 8, 12, 15, 16]+[21, 22, 23, 27, 30,31] + [55, 56, 58, 59, 60] + [75, 76, 78, 79, 80])
useful_trajectory_indexs_mask = np.array([0,1,2,3,4,5]+[6,7,8,9,10,11]+ [14,15,16,17,18,19] + [34,35,36,37,38] + [39,40,41,42,43])
negative_trajectory_indexs_mask = np.array([10,18])
active_joint_trajectory_700 = np.zeros((trajectory.shape[0],dof_muscle_700.shape[0]))
active_joint_trajectory_velocities_700 = np.zeros((trajectory_velocities.shape[0],dof_muscle_700.shape[0]))
whole_joint_trajectory = np.zeros((trajectory.shape[0],model.nq))
whole_joint_trajectory_velocities = np.zeros((trajectory_velocities.shape[0],model.nq))
# Create a MuJoCo viewer
viewer = mujoco.viewer.launch_passive(model, data)
i = 0
print(data.qpos.shape)
# Main simulation loop
while viewer.is_running() and i < trajectory.shape[0]:
    trajectory[i,negative_trajectory_indexs_mask] = -trajectory[i,negative_trajectory_indexs_mask]
    trajectory_velocities[i,negative_trajectory_indexs_mask] = -trajectory_velocities[i,negative_trajectory_indexs_mask]
    data.qpos = np.zeros_like(data.qpos)
    data.qpos[active_joint_indexs_700] = trajectory[i,useful_trajectory_indexs_mask]
    active_joint_trajectory_700[i] = data.qpos[dof_muscle_700]
    whole_joint_trajectory[i] = data.qpos.copy()
    data.qvel = np.zeros_like(data.qvel)
    data.qvel[active_joint_indexs_700] = trajectory_velocities[i,useful_trajectory_indexs_mask]
    active_joint_trajectory_velocities_700[i] = data.qvel[dof_muscle_700]
    whole_joint_trajectory_velocities[i] = data.qvel.copy()
    # p_ref = trajectory[i]
    # p_ref_pelvis = p_ref[:6]
    # p_ref_leg_r = p_ref[6:6+8]
    # p_ref_foot_r = np.zeros(len(dof_foot_r))
    # p_ref_leg_l = p_ref[14:14+8]
    # p_ref_foot_l = np.zeros(len(dof_foot_l))
    # p_ref_arm_r = np.concatenate((p_ref[34:34+5], np.zeros(2),), axis=0)
    # p_ref_arm_l = np.concatenate((p_ref[39:39+5], np.zeros(2),), axis=0)
    # p_ref_torso = p_ref[22:22+12]
    # data.qpos[dof_pelvis] = p_ref_pelvis
    # data.qpos[dof_leg_r] = p_ref_leg_r
    # data.qpos[dof_foot_r] = p_ref_foot_r
    # data.qpos[dof_leg_l] = p_ref_leg_l
    # data.qpos[dof_foot_l] = p_ref_foot_l
    # data.qpos[dof_arm_r] = p_ref_arm_r
    # data.qpos[dof_arm_l] = p_ref_arm_l
    # data.qpos[dof_torso] = p_ref_torso
    # data.qvel[:] = 0
    # other qpos except active_joint_indexs_700 should be 0
    # Step the simulation
    # mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    # Sleep for 0.002 seconds

    time.sleep(0.002)
    # Update the viewer
    viewer.sync()
    i=i+1

# Output active_joint_trajectory_700 as XML
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def trajectory_to_xml(trajectory,trajectory_velocities):
    root = ET.Element("mujoco")
    keyframe = ET.SubElement(root, "keyframe")
    
    for i, frame in enumerate(trajectory):
        key = ET.SubElement(keyframe, "key")
        key.set("name", f"walk_{i}")
        key.set("time", f"{i*0.002:.3f}")
        key.set("qpos", " ".join(map(str, frame)))
        key.set("qvel", " ".join(map(str, trajectory_velocities[i])))
    xml_str = ET.tostring(root, encoding="unicode")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
    
    with open("whole_joint_trajectory.xml", "w") as f:
        f.write(pretty_xml)

trajectory_to_xml(whole_joint_trajectory,whole_joint_trajectory_velocities)

# Close the viewer
viewer.close()

# ... rest of the existing code ...
