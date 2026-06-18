import numpy as np
import pandas as pd
import os

# Change the current working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def convert_qpos(qpos_file_path):
    """
    Parameters:
    qpos_file_path (str): Path to the input npy file.
    """
    """
    Convert a CSV file to a NumPy .npy file.

    Parameters:
    csv_file_path (str): Path to the input CSV file.
    """
    npy_file_path = qpos_file_path.replace('csv', 'npy')
    # Read the CSV file
    df = pd.read_csv(qpos_file_path)
    # Convert the DataFrame to a NumPy array
    np_array = df.to_numpy()
    # first column is time
    np_array = np_array[:, 1:]
    np_array = np_array[100:300, :]
    np_array[:, 2] -= 0.002
    # Save the NumPy array to a .npy file
    np.save(npy_file_path, np_array)
    
    print(f"Converted {qpos_file_path} to {npy_file_path}")

def convert_marker_xpos(marker_file_path):
    """
    Parameters:
    qpos_file_path (str): Path to the input npy file.
    """
    """
    Convert a CSV file to a NumPy .npy file.

    Parameters:
    csv_file_path (str): Path to the input CSV file.
    """
    npy_file_path = marker_file_path.replace('csv', 'npy')
    # Read the CSV file
    df = pd.read_csv(marker_file_path)
    # Convert the DataFrame to a NumPy array
    np_array = df.to_numpy()
    # first column is time
    np_array = np_array[:, 1:]
    np_array = np_array[100:300, :]
    # the axis 1 is x,y,z, I want another dim to restore x y z 
    np_array = np.reshape(np_array, (np_array.shape[0], -1, 3))
    np_array[:, :, 2] -= 0.002
    # Save the NumPy array to a .npy file
    np.save(npy_file_path, np_array)
    
    print(f"Converted {marker_file_path} to {npy_file_path}")

def csv_to_npy(csv_file_path):
    """
    Convert a CSV file to a NumPy .npy file.

    Parameters:
    csv_file_path (str): Path to the input CSV file.
    """
    npy_file_path = csv_file_path.replace('csv', 'npy')
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    # Convert the DataFrame to a NumPy array
    np_array = df.to_numpy()
    # first column is time
    np_array = np_array[:, 1:]
    np_array = np_array[100:300, :]
    # np_array[:, 2] -= 0.002
    # Save the NumPy array to a .npy file
    np.save(npy_file_path, np_array)
    
    print(f"Converted {csv_file_path} to {npy_file_path}")
    
def qpos_to_qvel(qpos_file_path):
    """
    Parameters:
    qpos_file_path (str): Path to the input npy file.
    """
    npy_file_path = qpos_file_path.replace('qpos', 'qvel')
    # Read the npy file
    qpos = np.load(qpos_file_path)
    # Compute the first order difference
    qvel = np.diff(qpos, axis=0)
    # insert the first qvel
    qvel = np.insert(qvel, 0, qvel[0], axis=0)
    # Save the NumPy array to a .npy file
    np.save(npy_file_path, qvel)
    print(f"Converted {qpos_file_path} to {npy_file_path}")
    
# Example usage:
# csv_to_npy('input.csv', 'output.npy')
if __name__ == "__main__":
    # csv_to_npy('../mice_motion_data/')
    convert_qpos('../ik_results/logs/joint_qpos.csv')
    qpos_to_qvel('../ik_results/logs/joint_qpos.npy')
    convert_marker_xpos('../ik_results/logs/marker_xpos.csv')
    
