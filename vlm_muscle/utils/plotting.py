import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_reward_weights(base_dir):
    """Load reward weights from JSON files in the specified directory."""
    # Find all stage folders and sort by stage number
    stage_dirs = sorted(
        [d for d in os.listdir(base_dir) if d.startswith("stage_")],
        key=lambda x: int(x.split("_")[1])
    )

    # Dictionary to store lists of values for each cost term and iterate over all stages
    all_terms = {}
    for stage_idx, stage_dir in enumerate(stage_dirs):
        json_path = os.path.join(base_dir, stage_dir, "cost_terms.json")
        with open(json_path, 'r') as f:
            cost_terms = json.load(f)

        # Add new terms with zeros for previous stages
        for term in cost_terms:
            if term not in all_terms:
                all_terms[term] = [None] * stage_idx

        # Ensure all existing terms have a placeholder for this stage
        for term in all_terms:
            if term in cost_terms:
                all_terms[term].append(cost_terms[term])
            else:
                all_terms[term].append(0)

    # Create DataFrame and save for later reference
    df = pd.DataFrame(all_terms, index=[f"Stage {i}" for i in range(len(stage_dirs))])
    df.to_json(os.path.join(base_dir, 'reward_weights.json'), orient='records', indent=4)
    return df

def plot_residual_heatmap(base_dir):
    """Generate a heatmap of residual weights across stages from JSON files."""
    reward_weights = load_reward_weights(base_dir)
    
    print(f"Plotting residual weights heatmap to {os.path.join(base_dir, 'plots', 'normalized_reward_weights.png')}")
    plt.figure(figsize=(15, 6))
    sns.heatmap(reward_weights, cmap="Blues")
    plt.title('Heatmap of Residual Weights Across Stages')
    plt.xlabel('Reward Features')
    plt.ylabel('Stage')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'plots', 'residual_heatmap.png'))
    plt.close()
    
def plot_normalized_area(base_dir):
    """Plot normalized area chart of reward weights."""
    reward_weights = load_reward_weights(base_dir)
    reward_weights_normalized = reward_weights.div(reward_weights.sum(axis=1), axis=0)
    
    plt.figure()
    reward_weights_normalized.plot.area(figsize=(10, 6), colormap="coolwarm", alpha=0.7)
    plt.title('Normalized Residual Feature Emphasis Over Stages')
    plt.xlabel('Stage')
    plt.ylabel('Proportion of Total Weight')
    plt.legend(title='Residual Features', loc='upper left', bbox_to_anchor=(-0.37, 1))
    plt.savefig(os.path.join(base_dir, 'plots', 'normalized_reward_weights.png'))
    plt.close()