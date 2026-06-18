import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_muscle_length(file_path, output_path):
    # Load data from CSV file
    import pandas as pd
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract column names
    column_names = df.columns.tolist()
    # Extract muscle names
    muscle_names = [col for col in column_names if not col.endswith('_target') and col != 'Time Step']
    
    print(f"Muscle names: {muscle_names}")
    print(f"Number of muscles: {len(muscle_names)}")
    # Separate actual and reference data
    actual_columns = [col for col in column_names if not col.endswith('_target') and col != 'Time Step']
    reference_columns = [col for col in column_names if col.endswith('_target')]
    
    # Extract data
    data_actual = df[actual_columns].values
    data_reference = df[reference_columns].values
    
    print(f"Loaded data shape: Actual {data_actual.shape}, Reference {data_reference.shape}")
    if data_actual.ndim != 2:
        raise ValueError("Unsupported data dimensions. Expected 2D array.")
    # Plot actual data as full line and reference data as dotted line
    # Determine the number of groups
    num_groups = data_actual.shape[1] // 10
    remaining = data_actual.shape[1] % 10

    # Create subplots for each group
    fig, axs = plt.subplots(num_groups + (1 if remaining > 0 else 0), 1, figsize=(12, 6*num_groups), sharex=True)
    if num_groups == 1:
        axs = [axs]

    for group in range(num_groups):
        start_idx = group * 10
        end_idx = start_idx + 10
        for i in range(start_idx, end_idx):
            muscle_name = muscle_names[i]
            axs[group].plot(data_actual[:, i], label=f'Actual {muscle_name}')
            axs[group].plot(data_reference[:, i], ':', label=f'Reference {muscle_name}')
        axs[group].set_title(f'Muscles {muscle_names[start_idx]}-{muscle_names[end_idx-1]}')
        axs[group].set_ylabel('Muscle Length')
        axs[group].legend()
        axs[group].grid(True)

    # Plot remaining muscles if any
    if remaining > 0:
        start_idx = num_groups * 10
        for i in range(start_idx, data_actual.shape[1]):
            muscle_name = muscle_names[i]
            axs[-1].plot(data_actual[:, i], label=f'Actual {muscle_name}')
            axs[-1].plot(data_reference[:, i], ':', label=f'Reference {muscle_name}')
        axs[-1].set_title(f'Muscles {muscle_names[start_idx]}-{muscle_names[-1]}')
        axs[-1].set_ylabel('Muscle Length')
        axs[-1].legend()
        axs[-1].grid(True)

    plt.tight_layout()
    plt.title('Muscle Lengths Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Muscle Length')
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory
    # Create a new figure for the difference plot
    fig_diff, axs_diff = plt.subplots(num_groups + (1 if remaining > 0 else 0), 1, figsize=(12, 6*num_groups), sharex=True)
    if num_groups == 1:
        axs_diff = [axs_diff]

    # Calculate the difference between target and actual
    data_diff = (data_reference - data_actual) * 1000

    for group in range(num_groups):
        start_idx = group * 10
        end_idx = start_idx + 10
        for i in range(start_idx, end_idx):
            muscle_name = muscle_names[i]
            axs_diff[group].plot(data_diff[:, i], label=f'{muscle_name}')
        axs_diff[group].set_title(f'Difference (Target - Actual) for Muscles {muscle_names[start_idx]}-{muscle_names[end_idx-1]}')
        axs_diff[group].set_ylabel('Difference in Muscle Length')
        axs_diff[group].set_ylim(-1, 1)  # Set y-axis range to (-1, 1)
        axs_diff[group].legend()
        axs_diff[group].grid(True)

    # Plot remaining muscles if any
    if remaining > 0:
        start_idx = num_groups * 10
        for i in range(start_idx, data_actual.shape[1]):
            muscle_name = muscle_names[i]
            axs_diff[-1].plot(data_diff[:, i], label=f'{muscle_name}')
        axs_diff[-1].set_title(f'Difference (Target - Actual) for Muscles {muscle_names[start_idx]}-{muscle_names[-1]}')
        axs_diff[-1].set_ylabel('Difference in Muscle Length')
        axs_diff[-1].set_ylim(-1, 1)  # Set y-axis range to (-1, 1)
        axs_diff[-1].legend()
        axs_diff[-1].grid(True)

    plt.tight_layout()
    plt.suptitle('Difference in Muscle Lengths (Target - Actual) Over Time')
    axs_diff[-1].set_xlabel('Time Step')

    # Save the difference figure
    diff_output_path = output_path.replace('.png', '_difference.png')
    plt.savefig(diff_output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

    print(f"Difference figure saved to {diff_output_path}")

    print(f"Figure saved to {output_path}")
    # Find the 5 muscles with the largest differences in the last frame
    last_frame_diff = np.abs(data_diff[-1, :])
    top_5_indices = np.argsort(last_frame_diff)[-5:][::-1]
    top_5_muscles = [muscle_names[i] for i in top_5_indices]
    top_5_differences = last_frame_diff[top_5_indices]

    print("\nTop 5 muscles with the largest differences in the last frame:")
    for muscle, diff in zip(top_5_muscles, top_5_differences):
        print(f"{muscle}: {diff:.4f} mm")

    # Create a bar plot for the top 5 differences
    plt.figure(figsize=(10, 6))
    plt.bar(top_5_muscles, top_5_differences)
    plt.title("Top 5 Muscles with Largest Differences in Last Frame")
    plt.xlabel("Muscle Name")
    plt.ylabel("Absolute Difference (mm)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the bar plot
    bar_plot_path = output_path.replace('.png', '_top5_diff.png')
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Top 5 differences bar plot saved to {bar_plot_path}")

def main():
    import os
    parser = argparse.ArgumentParser(description="Plot muscle length data from an .npy file")
    # -f is the input file path
    parser.add_argument("-f", "--input_file", help="Path to the input .npy file", default='/home/zsn/research/mice_training/ms_envs/msmodel_gym/envs/logs/muscle_lengths_20241018_143552.csv')
    # output_file_path is in the same directory as the input_file
    args = parser.parse_args()
    output_file_path = os.path.join(os.path.dirname(args.input_file), "muscle_length_plot.png")
    plot_muscle_length(args.input_file, output_file_path)

if __name__ == "__main__":
    main()

