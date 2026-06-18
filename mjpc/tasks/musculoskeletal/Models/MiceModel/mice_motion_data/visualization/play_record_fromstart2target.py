import mujoco
import os
import numpy as np
import time
import imageio
import matplotlib.pyplot as plt
import csv

# ... existing imports ...

os.chdir(os.path.dirname(os.path.abspath(__file__)))

### Initialize model and Read joint trajectory
mujoco_model_path = '../../CyberMice_Hanging.xml'
model = mujoco.MjModel.from_xml_path(mujoco_model_path)
data = mujoco.MjData(model)
qpos_file_path = '../ik_results/logs/combined_qpos.npy'
qpos_array = np.load(qpos_file_path)

print("qpos_array.shape: ", qpos_array.shape)

# Separate start and target qpos
num_frames = qpos_array.shape[0]
start_qpos = qpos_array[:num_frames // 2]
target_qpos = qpos_array[num_frames // 2:]

# Create interpolated qpos array
num_interpolation_steps = 100  # You can adjust this value
interpolated_qpos = np.zeros((num_interpolation_steps, start_qpos.shape[0]))

for i in range(num_interpolation_steps):
    t = i / (num_interpolation_steps - 1)
    interpolated_qpos[i] = (1 - t) * start_qpos + t * target_qpos

### Main simulation loop
i = 0
data.qpos = interpolated_qpos[0]
mujoco.mj_fwdPosition(model, data)

frames_list = []
muscle_length_history = []

rgb_renderer = mujoco.Renderer(model, width=1302, height=1080)
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True
time_step = 1

while i < interpolated_qpos.shape[0]:
    # Set joint positions
    data.qpos = interpolated_qpos[i]
    data.qvel = np.zeros_like(data.qvel)
    mujoco.mj_fwdPosition(model, data)

    # Record muscle lengths
    muscle_length_history.append(data.actuator_length.copy())

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

# Save muscle length history
muscle_names = [model.actuator(i).name for i in range(model.nu)]
log_dir = '../ik_results/logs'
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, f'muscle_lengths_{time_str}.csv')

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time Step'] + muscle_names)
    for i, lengths in enumerate(muscle_length_history):
        writer.writerow([i] + lengths.tolist())

print(f"Muscle length data saved to {csv_path}")

# Plot muscle lengths
def plot_muscle_length(data, muscle_names, output_path):
    num_muscles = len(muscle_names)
    num_groups = num_muscles // 10
    remaining = num_muscles % 10

    # Calculate delta length change
    delta_data = data - data[0]  # Subtract the initial length from all frames

    # Plot absolute lengths
    fig, axs = plt.subplots(num_groups + (1 if remaining > 0 else 0), 1, figsize=(12, 6*num_groups), sharex=True)
    if num_groups == 1:
        axs = [axs]

    for group in range(num_groups):
        start_idx = group * 10
        end_idx = start_idx + 10
        for i in range(start_idx, end_idx):
            axs[group].plot(data[:, i], label=muscle_names[i])
        axs[group].set_title(f'Muscles {muscle_names[start_idx]}-{muscle_names[end_idx-1]}')
        axs[group].set_ylabel('Muscle Length')
        axs[group].legend()
        axs[group].grid(True)

    if remaining > 0:
        start_idx = num_groups * 10
        for i in range(start_idx, num_muscles):
            axs[-1].plot(data[:, i], label=muscle_names[i])
        axs[-1].set_title(f'Muscles {muscle_names[start_idx]}-{muscle_names[-1]}')
        axs[-1].set_ylabel('Muscle Length')
        axs[-1].legend()
        axs[-1].grid(True)

    plt.tight_layout()
    plt.suptitle('Absolute Muscle Lengths Over Time')
    axs[-1].set_xlabel('Time Step')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Absolute muscle length plot saved to {output_path}")

    # Plot delta length change
    fig, axs = plt.subplots(num_groups + (1 if remaining > 0 else 0), 1, figsize=(12, 6*num_groups), sharex=True)
    if num_groups == 1:
        axs = [axs]

    for group in range(num_groups):
        start_idx = group * 10
        end_idx = start_idx + 10
        for i in range(start_idx, end_idx):
            axs[group].plot(delta_data[:, i], label=muscle_names[i])
        axs[group].set_title(f'Muscles {muscle_names[start_idx]}-{muscle_names[end_idx-1]}')
        axs[group].set_ylabel('Delta Muscle Length')
        axs[group].legend()
        axs[group].grid(True)

    if remaining > 0:
        start_idx = num_groups * 10
        for i in range(start_idx, num_muscles):
            axs[-1].plot(delta_data[:, i], label=muscle_names[i])
        axs[-1].set_title(f'Muscles {muscle_names[start_idx]}-{muscle_names[-1]}')
        axs[-1].set_ylabel('Delta Muscle Length')
        axs[-1].legend()
        axs[-1].grid(True)

    plt.tight_layout()
    plt.suptitle('Delta Muscle Lengths Over Time')
    axs[-1].set_xlabel('Time Step')

    delta_output_path = output_path.replace('.png', '_delta.png')
    plt.savefig(delta_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Delta muscle length plot saved to {delta_output_path}")

# Plot muscle lengths
muscle_length_data = np.array(muscle_length_history)
plot_path = os.path.join(log_dir, f'muscle_length_plot_{time_str}.png')
plot_muscle_length(muscle_length_data, muscle_names, plot_path)
