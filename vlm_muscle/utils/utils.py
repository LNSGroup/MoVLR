import os
import re
import ast
import cv2
import json
import logging
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
from scipy.stats import linregress

from typing import Optional

def run_bash(file_path, arguments=[]):
    run_cmd = ["bash", file_path] + arguments
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    
    return result

def get_recording(model_path, data_path, output_dir, fps, output_prefix, camera, bash_script):
    arguments = [model_path, data_path, output_dir, camera, fps, output_prefix]
    
    result = run_bash(bash_script, arguments)
    if result.returncode != 0:
        print(f"Error running bash script: {result.stderr}")
        return None
    
    print("Bash script executed successfully.")
    print("Output:", result.stdout)
    
    return result

def get_image(video_path, nimages):    
    # Get output directory
    out_dir = os.path.dirname(video_path)
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if nimages > total_frames:
        nimages = total_frames
    
    if nimages == 1:
        indices = [0]
    else:
        indices = [int(round(i * (total_frames - 1) / (nimages - 1))) for i in range(nimages)]
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            img_path = os.path.join(out_dir, f"image{saved+1}.png")
            cv2.imwrite(img_path, frame)
            saved += 1
    cap.release()
    print(f"Saved {saved} images to {out_dir}")

def get_latest_txt_file(directory: str) -> Optional[str]:
    txt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    if not txt_files:
        return None
    latest_file = max(txt_files, key=os.path.getmtime)
    return latest_file

def file_to_string(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
def extract_code(input_string):
    # Regex patterns to extract code enclosed in various markers
    patterns = [
        r'```python(.*?)```',   # Python code block
        r'```cpp(.*?)```',      # C++ code block
        r'```xml(.*?)```',      # XML code block
        r'```(.*?)```',         # Generic code block
        r'"""(.*?)"""',         # Triple-quote block
        r'""(.*?)""'            # Double-quote block
    ]
    code_string = None
    for pattern in patterns:
        code_string = re.search(pattern, input_string, re.DOTALL|re.IGNORECASE)
        if code_string is not None:
            code_string = code_string.group(1).strip()
            break
        
    code_string = input_string if not code_string else code_string
    return code_string

def extract_cost(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    ret_cost = ""
    for sensor in root.findall("sensor"):
        for user in sensor.findall("user"):
            name = user.get("name")
            user_attr = user.get("user")
            
            ret_cost += f"{name}: {ast.literal_eval("[" + user_attr.replace(" ", ",") + "]")[1]}\n"

    return ret_cost

def get_sensors(file):
    # Prefixes to search for
    prefixes = ("<framepos", "<subtreecom", "<subtreelinvel", "<frameyaxis", "<framexaxis", "<framequat")

    # Collect matches
    ret_sensors = ""
    for line in file:
        stripped = line.strip()
        if stripped.startswith(prefixes):
            ret_sensors += stripped + "\n"
                
    return ret_sensors

def create_logger(log_filename):
    # Remove any existing handlers to prevent duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(filename=log_filename,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filemode='w', level=logging.INFO)
    logger = logging.getLogger()
    
    return logger

def write_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)
     
def get_file_ending(file_dir, ending):
    data_file = None
    for file in os.listdir(file_dir):
        if file.endswith(ending):
            data_file = os.path.join(file_dir, file)
            break
    if data_file is None:
        raise FileNotFoundError(f"No file ending with {ending} found in the directory.")     
    
    return data_file

def get_cost_terms(input_file, output_file):
    # Load and parse the XML
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Initialize dictionary to hold cost terms
    cost_dict = {}

    # Find the <sensor> element
    sensor_elem = root.find('sensor')

    # Loop through all <user> tags in <sensor>
    for user_elem in sensor_elem.findall('user'):
        name = user_elem.get('name')
        user_data = user_elem.get('user')
        if user_data:
            user_vals = user_data.split()
            if len(user_vals) > 1:
                try:
                    cost_val = float(user_vals[1])
                    cost_dict[name] = int(cost_val)
                except ValueError:
                    pass  # skip if it's not a number
    
    with open(output_file, 'w') as f:
        json.dump(cost_dict, f, indent=4)

def compress_residual_history(input_file, output_file):
    # Load the JSON
    with open(input_file, "r") as f:
        data = json.load(f)

    # Case 1: dict of {"Stage N": {...}}
    if isinstance(data, dict):
        stages = sorted(data.keys(), key=lambda x: int(x.split()[1]))
        stage_items = [data[s] for s in stages]

    # Case 2: list of stage dicts
    elif isinstance(data, list):
        stage_items = data
        stages = list(range(1, len(stage_items) + 1))

    else:
        raise ValueError("Unexpected JSON structure: must be dict or list")

    # Collect all unique terms
    all_terms = set()
    for weights in stage_items:
        all_terms.update(weights.keys())

    # Build matrix-like dict
    matrix_repr = {}
    for term in sorted(all_terms):
        values = []
        for weights in stage_items:
            values.append(weights.get(term, 0))  # fill missing with 0
        matrix_repr[term] = values

    # Save compact JSON (one line)
    with open(output_file, "w") as f:
        json.dump(matrix_repr, f, separators=(",", ":"))

    print(f"Matrix-style JSON saved to {output_file} (one-line format)")
    
def update_task_model(xml_path, task_name, output_path=None):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find the <include> tag that points to mj_FullBody
    for include in root.findall("include"):
        if "mj_FullBody" in include.attrib["file"]:
            include.set("file", f"../Models/mj_FullBody/{task_name}.xml")
            break
    
    # Write back to file (in-place or to new file)
    if output_path is None:
        output_path = xml_path
    tree.write(output_path)
    
def get_pelvis_joints(input_file, output_file):
    # Process the file
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line by commas and take the first six items
            first_six = line.strip().split(',')[:6]
            # Join them back into a comma-separated string
            new_line = ','.join(first_six)
            # Write to the output file
            outfile.write(new_line + '\n')