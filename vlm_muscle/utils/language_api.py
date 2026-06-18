import os
import re
import argparse
import dashscope
from dashscope import MultiModalConversation
from dashscope import Generation

# Setup API key
DEBUG = False
API_KEY = os.environ.get("QWEN_API_KEY")
if not API_KEY:
    raise ValueError("Please set the API_KEY environment variable.")
dashscope.api_key = API_KEY

PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(BOX_TAG_PATTERN, '', text)

def image_api(image_paths, prompt):
    content = []
    if isinstance(image_paths, list):
        for image in image_paths:
            assert os.path.exists(image), f"Video path does not exist: {image}"
            content_input = {"type": "image", "image": f"file://{image}"}
    else:
        assert os.path.exists(image_paths), f"Video path does not exist: {image_paths}"
        content.append({'video': f'file://{image_paths}'})

    content.append({'text': prompt})
    
    messages = [{'role': 'user', 'content': content}]
    print("Sending video request to Qwen2.5-VL-72B-Instruct")
    responses = MultiModalConversation.call(model='qwen2.5-vl-72b-instruct',
                                            messages=messages,
                                            temperature=1.0)
    print(responses)
    full_response = responses["output"]["choices"][0]["message"]["content"][0]["text"]

    if DEBUG:
        print("\n========== Model Output ==========\n")
        print(_remove_image_special(full_response).strip())
        print("\n==================================")

    return _remove_image_special(full_response).strip()

    
def vlm_api(video_paths, prompt):
    content = []
    if isinstance(video_paths, list):
        print("Multiple video paths detected.")
        for video in video_paths:
            assert os.path.exists(video), f"Video path does not exist: {video}"
            content.append({"type": "video", "video": f"file://{video}"})
    else:
        assert os.path.exists(video_paths), f"Video path does not exist: {video_paths}"
        content.append({'video': f'file://{video_paths}'})

    content.append({'text': prompt})
    
    print(content)
    messages = [{'role': 'user', 'content': content}]
    print("Sending video request to Qwen2.5-VL-72B-Instruct")
    responses = MultiModalConversation.call(model='qwen2.5-vl-72b-instruct',
                                            messages=messages,
                                            temperature=1.0)
    # print(responses)
    print("Tokens used:", responses.get("usage", {}).get("total_tokens"))
    full_response = responses["output"]["choices"][0]["message"]["content"][0]["text"]

    if DEBUG:
        print("\n========== Model Output ==========\n")
        print(_remove_image_special(full_response).strip())
        print("\n==================================")

    return _remove_image_special(full_response).strip()

def llm_api(prompt):
    content = []
    content.append({'text': prompt})
    
    messages = [{'role': 'user', 'content': content}]
    
    print("Sending request to Qwen 2.5 Coder 32B Instruct")
    responses = Generation.call(model='qwen2.5-coder-32b-instruct',
                                messages=messages,
                                temperature=1.0)

    print("Tokens used:", responses.get("usage", {}).get("total_tokens"))
    full_response = responses["output"]["text"]

    if DEBUG:
        print("\n========== Model Output ==========\n")
        print(_remove_image_special(full_response).strip())
        print("\n==================================")

    return _remove_image_special(full_response).strip()
    
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-32B-Instruct Inference Script")
    parser.add_argument("--video", type=str, help="Path to the input video file")
    parser.add_argument("--prompt", type=str, required=True, help="Text file including prompt to send to the model")
    
    return parser.parse_args()

def main():
    args = parse_args()

    task = "walk in a straight line with a steady pace and proper posture"
    run_inference(args.prompt, task)

if __name__ == "__main__":
    main()
