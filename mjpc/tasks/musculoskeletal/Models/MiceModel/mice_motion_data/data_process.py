import os
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

# get the file path of the file directory
file_path = os.path.dirname(__file__)
data_name = 'data/2020_12_22_1.h5'
data_path = os.path.join(file_path, data_name)
file = h5py.File(data_path, 'r')

# List all groups and datasets in the file
print("Keys: %s" % file.keys())     # Keys: <KeysViewHDF5 ['behavior', 'ephys', 'pose']>

group_behavior = file['behavior']
group_ephys = file['ephys']
group_pose = file['pose']

# List all datasets in the group
print("Datasets in group behavior: %s" % group_behavior.keys())     # <KeysViewHDF5 ['motion_mapper']>
print("Datasets in group ephys: %s" % group_ephys.keys())           # <KeysViewHDF5 ['spike_counts']>
print("Datasets in group pose: %s" % group_pose.keys())             # <KeysViewHDF5 ['keypoints', 'qpos']>

# Read the dataset
dataset_motion_mapper = group_behavior['motion_mapper']     # (360000,)
dataset_spike_counts = group_ephys['spike_counts']          # (360000, 131)
dataset_keypoints = group_pose['keypoints']                 # (360000, 3, 23), 23 keypoints
dataset_qpos = group_pose['qpos']                           # (360000, 74), 74 DoF

# sampled at 50 Hz, 360000 frames = 2 hours
time_length = 60  # seconds
sample_rate = 50  # Hz

fig = plt.figure(figsize=(12, 8))

# Initialize subplots
ax1 = fig.add_subplot(232, projection='3d')
ax2 = fig.add_subplot(234, projection='3d')
ax3 = fig.add_subplot(235, projection='3d')
ax4 = fig.add_subplot(236, projection='3d')

num_frames = time_length * sample_rate

# check result dir
result_dir = os.path.join(file_path, 'result')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

def update_frame(num, data, scat1, scat2, scat3, scat4, texts1, texts2, texts3, texts4):
    for scat, texts in zip([scat1, scat2, scat3, scat4], [texts1, texts2, texts3, texts4]):
        scat._offsets3d = (data[num, 0, :], data[num, 1, :], data[num, 2, :])
        for i, text in enumerate(texts):
            text.set_position((data[num, 0, i], data[num, 1, i]))
            text.set_3d_properties(data[num, 2, i], 'z')

    return scat1, scat2, scat3, scat4, texts1, texts2, texts3, texts4

data = dataset_keypoints[:num_frames]

# Initialize scatter plots
scat1 = ax1.scatter(data[0, 0, :], data[0, 1, :], data[0, 2, :])
scat2 = ax2.scatter(data[0, 0, :], data[0, 1, :], data[0, 2, :])
scat3 = ax3.scatter(data[0, 0, :], data[0, 1, :], data[0, 2, :])
scat4 = ax4.scatter(data[0, 0, :], data[0, 1, :], data[0, 2, :])

# Add labels to the keypoints with smaller font size
keypoint_labels = [f'P{i}' for i in range(data.shape[2])]
texts1 = [ax1.text(data[0, 0, i], data[0, 1, i], data[0, 2, i], keypoint_labels[i], fontsize=8) for i in range(data.shape[2])]
texts2 = [ax2.text(data[0, 0, i], data[0, 1, i], data[0, 2, i], keypoint_labels[i], fontsize=8) for i in range(data.shape[2])]
texts3 = [ax3.text(data[0, 0, i], data[0, 1, i], data[0, 2, i], keypoint_labels[i], fontsize=8) for i in range(data.shape[2])]
texts4 = [ax4.text(data[0, 0, i], data[0, 1, i], data[0, 2, i], keypoint_labels[i], fontsize=8) for i in range(data.shape[2])]

# Set axis labels
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

# Set subplot titles
ax1.set_title('Initial View')
ax2.set_title('Front View')
ax3.set_title('Left View')
ax4.set_title('Top View')

# Adjust views
ax2.view_init(elev=0, azim=0)   # Front view
ax3.view_init(elev=0, azim=90)  # Left view
ax4.view_init(elev=90, azim=-90) # Top view

for i in tqdm(range(num_frames), desc='Generating frames'):
    update_frame(i, data, scat1, scat2, scat3, scat4, texts1, texts2, texts3, texts4)
    # plt.show()
    plt.savefig(os.path.join(result_dir, f'frame_{i:04d}.jpg'))

# Create a list of image file paths
image_files = [os.path.join(result_dir, f'frame_{i:04d}.jpg') for i in range(num_frames)]

# Create a video clip from the image sequence
clip = ImageSequenceClip(image_files, fps=sample_rate)

# Write the video file
video_path = os.path.join(result_dir, 'animation.mp4')
clip.write_videofile(video_path, codec='libx264')

# Remove the image files after creating the video
for filename in image_files:
    os.remove(filename)

# Close the file
file.close()