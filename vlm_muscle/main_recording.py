import os
import copy
import time
import json
import argparse
from utils.build_cpp_utils import build_project
from utils.utils import write_file, file_to_string, get_file_ending, get_recording
 
def parse_args():
    parser = argparse.ArgumentParser()

    # Add arguments with default values from config
    parser.add_argument('--config_file', '-f', type=str, default=None, help='Path to the configuration file')
    args = parser.parse_args()

    # Load yaml config file
    config = json.load(open(args.config_file, 'r'))
    with open(args.config_file, 'r') as f:
        config_str = f.read()  
        
    arg_config = argparse.Namespace(**config)
    setattr(arg_config, 'config_file', args.config_file)
    return arg_config

def main():
    args = parse_args()
    
    # Setup log directories
    curr_path  = os.path.dirname(__file__)
    log_dir    = os.path.join(curr_path, "logs", args.body, args.task_name)
    output_dir = os.path.join(log_dir, time.strftime("%Y%m%d_%H%M%S"))
    model_dir  = os.path.abspath(os.path.join(curr_path, ".."))
    print("output dir", output_dir)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(model_dir, args.model_path)
    write_file(os.path.join(output_dir, "config.json"), json.dumps(vars(args), indent=4))
    
    # Read in the task and environment files, and write to the build directory
    task_file, env_file = file_to_string(args.task_path), file_to_string(args.residual_path) # files to run
    task_file_orig, env_file_orig = file_to_string(args.build_task_path), file_to_string(args.build_env_path) # original files in build directory 
    
    write_file(args.build_task_path, task_file)
    write_file(args.build_env_path, env_file)    
    
    # Build the project, save task/env files to log, and record
    build_project(args)
    data_file = get_file_ending(output_dir, '_qpos.txt')
    print(data_file)
    get_recording(model_path, data_file, output_dir, "500", f"output_500", os.path.join(curr_path, args.video_script))
    get_recording(model_path, data_file, output_dir, "250", f"output_250", os.path.join(curr_path, args.video_script))
    
    write_file(os.path.join(output_dir, "task.xml"), task_file)
    write_file(os.path.join(output_dir, "task.xml"), task_file)
    
    # Revert the environment and task files to original
    write_file(env_path, env_file_orig)
    write_file(task_path, task_file_orig)
     
if __name__=="__main__":
    main()