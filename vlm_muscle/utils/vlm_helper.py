import os
import torch
import subprocess
from openai import OpenAI

from google import genai
from google.genai.types import HttpOptions, Part

# TODO the current implementation assumes that the remote host is set up with ssh keys and has alias
def run_remote(prompt, conda_env, remote_host, remote_folder, remote_script_path, cuda_devices="1,2,3", llm="qwen", video_path=None):
    """Helper function to run Qwen VLM on a remote host via SSH"""
    enc_cmd = 'export PYTHONIOENCODING=utf-8'
    hf_cmd = 'export HF_ENDPOINT=https://hf-mirror.com'
    conda_cmd = f'source ~/anaconda3/etc/profile.d/conda.sh && conda activate {conda_env} && echo $CONDA_DEFAULT_ENV'
    
    print("video path", video_path)
    if video_path is None:
        # Compose SSH command to run the remote Python script
        python_command = f"python {os.path.join(remote_folder, remote_script_path)} --llm {llm} --prompt \"{prompt}\""
        remote_command = (f"{enc_cmd} && {hf_cmd} && {conda_cmd} && {python_command}")
    else:
        # Upload the video video file to the remote host
        # scp -P 2205 /path/to/local/video.mp4 username@remote.server:/tmp/video.mp4
        upload_command = [
            f"scp -P 2205 {video_path} {remote_host}:{remote_folder}videos"
        ]
        subprocess.run(upload_command, shell=True, check=True)
        # Compose SSH command to run the remote Python script without video
        python_command = f"CUDA_VISIBLE_DEVICES={cuda_devices} python {os.path.join(remote_folder, remote_script_path)} --prompt \"{prompt}\" --video_path {remote_folder}videos/{video_path.split('/')[-1]}"
        remote_command = (f"{enc_cmd} && {hf_cmd} && {conda_cmd} && {python_command}")
        print("remote command", remote_command)
    
    ssh_command = ["ssh", f"{remote_host}", remote_command]
    
    # Run the ssh command and capture output
    result = subprocess.run(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        print(f"Error running remote script: {result.stderr}")
        return None
    
    # result.stdout contains everything printed by the remote script
    return result.stdout

def deepseek_query(prompt, system=None, model="deepseek-reasoner", max_completion_tokens=32768):
    assert isinstance(prompt, str), "Prompt must be a string."
    assert isinstance(max_completion_tokens, int) and max_completion_tokens > 0, "max_completion_tokens must be a positive integer."
    assert torch.cuda.is_available(), "CUDA is not available. Ensure you have a compatible GPU and PyTorch installed with CUDA support"

    client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")

    if system is None:
        system = "You are a helpful assistant."
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        max_completion_tokens=max_completion_tokens,
    )

    return response.choices[0].message.content

def gemini_api(vid_dir, prompt) -> str:
    # [START googlegenaisdk_textgen_with_local_video]

    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    model_id = "gemini-2.0-flash"

    # Read local video file content
    with open(vid_dir, "rb") as fp:
        # Video source: https://storage.googleapis.com/cloud-samples-data/generative-ai/video/describe_video_content.mp4
        video_content = fp.read()

    response = client.models.generate_content(
        model=model_id,
        contents=[
            prompt,
            Part.from_bytes(data=video_content, mime_type="video/mp4"),
        ],
    )

    return response.text

if __name__ == "__main__":
    # Example usage
    conda_env = "dynsyn"
    prompt = """The video depicts a muscle skeleton model. Describe what is happening in the video, and also describe what happens to the muscle skeleton model at the end"""
    remote_host = "lnsgroup"
    remote_user = "saraswati"
    folder = "/home/saraswati/Desktop/grad/llm_helpers/"
    remote_script_path = f"{folder}run_llm.py"
    video_path = "/home/cxy/Desktop/mpc2/vlm_muscle/logs/20250623_225524_rec/output_video_final.mp4"

    output = run_remote(prompt, conda_env, remote_host, folder, remote_script_path, video_path=video_path)

    print("Captured remote output:")
    print(output)