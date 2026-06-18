import os
import re
import tqdm
import copy
import time
import json
import shutil
import logging
import argparse
import traceback
from pathlib import Path
from utils.utils import *
from utils.build_cpp_utils import build_project
from utils.vlm_helper import run_remote, deepseek_query
from utils.language_api import vlm_api, llm_api
from utils.metrics import calculate_success
from utils.plotting import plot_residual_heatmap, plot_normalized_area
    
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

def run(args):
    # Setup log directories
    species      = "humanoid" if args.body.split(" ")[0] != "ostrich" else ""
    curr_path    = os.path.dirname(__file__)
    env_path     = os.path.abspath(os.path.join(curr_path, "..", args.residual_path))
    task_path    = os.path.abspath(os.path.join(curr_path, "..", args.task_path))
    main_path    = os.path.abspath(os.path.join(curr_path, "..", args.main_path))
    log_dir      = os.path.join(curr_path, "logs", args.body.split(" ")[0] , args.task_name)
    output_dir   = os.path.join(log_dir, time.strftime("%Y%m%d_%H%M%S"))
    model_dir    = os.path.abspath(os.path.join(curr_path, ".."))
    plot_dir     = os.path.join(output_dir, "plots")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    model_path = os.path.join(model_dir, args.model_path)
    write_file(os.path.join(output_dir, "config.json"), json.dumps(vars(args), indent=4))
    
    # Read in original environment files
    env_file_orig, task_file_orig = file_to_string(env_path), file_to_string(task_path)
    env_file, task_file = copy.deepcopy(env_file_orig), copy.deepcopy(task_file_orig)
    new_task = copy.deepcopy(task_file)
    
    # Read in files to change based on body part and task
    main_orig = file_to_string(main_path)
    main_file = copy.deepcopy(main_orig)
    
    # Read in the prompt files
    try:
        operators    = file_to_string(os.path.abspath(os.path.join(os.path.dirname(__file__), "./prompts/operators.txt")))
        code_tips    = file_to_string(os.path.abspath(os.path.join(os.path.dirname(__file__), "./prompts/code_output_tip.txt")))
    except FileNotFoundError:
        print("Prompt files not found")
    
    # Update the main.cc file based on specific task
    pattern = r'ABSL_FLAG\(\s*std::string,\s*task,\s*".*?",\s*"Which model to load on startup."\s*\);'
    replacement = f'ABSL_FLAG(std::string, task, "{args.task}", "Which model to load on startup.");'
    
    new_main, n = re.subn(pattern, replacement, main_file)
    write_file(main_path, new_main)
        
    try:        
        for stage in tqdm.tqdm(range(args.stages)):
            output_dir_stage = os.path.join(output_dir, f"stage_{stage}")
            output_dir_prev  = os.path.join(output_dir, f"stage_{stage-1}")
            os.makedirs(output_dir_stage, exist_ok=True)
            os.chdir(output_dir_stage)
            
            prev_task = os.path.join(output_dir_prev, "task.xml")
            
            # For each new stage copy the new code into the environment variable

            if stage > 1:
                env_file = new_code        
            
            attempt = 0
            for attempt in range(args.nattempts):
                try:
                    # Create new logger file for each stage
                    logger = create_logger(os.path.join(output_dir_stage, f"stage{stage}_{attempt}_logs.log"))
                    logger.info(f"Stage: {stage}/{args.stages}, Attempt: {attempt+1}/{args.nattempts}")
                    
                    if stage != 0:
                        residual_weights = file_to_string(os.path.join(output_dir, 'compressed_weights.txt'))

                        # First use the vlm to analyze the video and get corresponding feedback
                        video_path = os.path.join(output_dir_prev, "output_125_video_final.mp4")
                        if not Path(video_path).is_file() and stage != 1: # if the video file does not exist, copy the previous stage's video file
                            logger.error(f"Video file {video_path} does not exist, copying from previous stage")
                            dir_2prev = os.path.join(output_dir, f"stage_{stage-2}") # output direction from 2 stages before
                            for file_name in os.listdir(dir_2prev):
                                src_path = os.path.join(dir_2prev, file_name)
                                dst_path = os.path.join(output_dir_prev, file_name)
                                if not file_name.endswith('.log'):
                                    shutil.copy2(src_path, dst_path)
                                    
                        prev_rewards = extract_cost(prev_task)
                        focus        = "fullbody" if f"{args.body.split(" ")[0]}" in ["ostrich", "fullbody"] else "arm"
                        video_prompt = file_to_string(os.path.abspath(os.path.join(os.path.dirname(__file__), "./prompts/video_prompt.txt"))).format(reward_terms=prev_rewards, task=args.task_description, body=f"{args.body.split(" ")[0]} {species}", focus=focus)
                        video_analysis = vlm_api(video_path, video_prompt)
                        
                        # Use the feedback from the LLM to edit the environment file
                        logger.info(f"Video Feedback: {video_analysis}")
                        code_prompt = file_to_string(os.path.abspath(os.path.join(os.path.dirname(__file__), "./prompts/env_prompt.txt"))).format(task=args.task_description, env_code=env_file, task_code=new_task, feedback_string=video_analysis, operations=operators, code_tips=code_tips, residual_terms=residual_weights)
                        
                        new_code = llm_api(code_prompt)
                        logger.info(f"New Code: {new_code}")
                        new_code = extract_code(new_code)
                        
                        print("New environment code received, creating new task file")
                        
                        # Use the new code to edit the task file
                        task_prompt = file_to_string(os.path.abspath(os.path.join(os.path.dirname(__file__), "./prompts/task_prompt.txt"))).format(task=args.task_description, env_code=new_code, task_code=new_task, feedback_string=video_analysis, residual_terms=residual_weights, body=f"{args.body.split(" ")[0]} {species}")
                        new_task = llm_api(task_prompt)
                        logger.info(f"New Task: {new_task}")
                        new_task = extract_code(new_task)
                        
                        print("New task file created, updating environment and task files")
                        
                        # Update the environment and task files with the new env/task code
                        write_file(env_path, new_code)
                        write_file(task_path, new_task)
                    
                        write_file(os.path.join(output_dir_stage, args.residual_path.split('/')[-1]), new_code)
                        write_file(os.path.join(output_dir_stage, args.task_path.split('/')[-1]), new_task)
                        get_cost_terms(os.path.join(output_dir_stage, args.task_path.split('/')[-1]), os.path.join(output_dir_stage, 'cost_terms.json'))
                    else:
                        logger.info(f"Initial environment file: \n{env_file_orig}")
                        logger.info(f"Initial task file: \n{task_file_orig}")
                        
                        write_file(os.path.join(output_dir_stage, args.residual_path.split('/')[-1]), env_file_orig)
                        write_file(os.path.join(output_dir_stage, args.task_path.split('/')[-1]), task_file_orig)
                        get_cost_terms(os.path.join(output_dir_stage, args.task_path.split('/')[-1]), os.path.join(output_dir_stage, 'cost_terms.json'))
                        
                    # Build the project and save/retrieve corresponding data files
                    start = time.time()
                    build_project(args)
                    end = time.time()
                    print(f"Total build time: {end - start:.2f} seconds")
                    data_file = get_file_ending(output_dir_stage, '_qpos.txt')
                    print(data_file)
                    # Record the corresponding muscle skeleton motion, and save a copy of the environment and task files in logs folder
                    logger.info(f"Recording for stage {stage} with data file: {data_file}")
                    get_recording(model_path, data_file, output_dir_stage, "500", f"output_500", args.camera, os.path.join(curr_path, args.video_script))
                    get_recording(model_path, data_file, output_dir_stage, "125", f"output_125", args.camera, os.path.join(curr_path, args.video_script))
                    
                    break
                except Exception as e:
                    logger.error(f"Error in stage {stage}, attempt {attempt}: {e}")
                    traceback.print_exc()
                    if attempt == args.nattempts - 1:
                        logger.error("Max attempts reached. Reverting all file and exiting.")
                        write_file(env_path, env_file_orig)
                        write_file(task_path, task_file_orig)
                        break
                    else:
                        logger.info("Retrying...")
                        time.sleep(5)
                        continue
                    
            # Plot residual weights and normalized area chart for each stage
            plot_residual_heatmap(output_dir)
            plot_normalized_area(output_dir)
            compress_residual_history(os.path.join(output_dir, 'reward_weights.json'), os.path.join(output_dir, 'compressed_weights.txt'))
        
        # At the end of all stages, replace the edited environments with the original ones saved above
        write_file(env_path, env_file_orig)
        write_file(task_path, task_file_orig)
        write_file(main_path, main_orig)
        logger.info("All stages completed successfully.")
    except KeyboardInterrupt:
        print("Process interrupted by user. Reverting files to original state.")
        write_file(env_path, env_file_orig)
        write_file(task_path, task_file_orig)
        write_file(main_path, main_orig)

def main():
    args = parse_args()
    run(args)
     
if __name__=="__main__":
    main()