import base64
from openai import OpenAI
import os
import json
from omegaconf import OmegaConf  
description_prompt_mapping = {
    "generation": '**Role:**\nYou are a senior Frontend Product Manager and UI Designer. Please analyze the provided webpage screenshot to generate a concise requirement description for frontend development.\n\n**Task:**\nAnalyze the provided screenshot and summarize the webpage\'s core visual characteristics and key interactions using **very concise natural language (under 150 words)**.\n\n**Strict Rules:**\n- **Zero Code Principle:** Do NOT use any code snippets, HTML tags, or technical CSS parameters.\n- **Visuals Only:** Describe only what a user can "see" and "feel." For example, instead of saying "set border-radius to 50%," say "the avatar is displayed as a perfect circle."\n- **Macro Perspective:** Focus on the overall layout (Header/Body/Footer), the primary color palette/mood, the visual style, and the core functional components (e.g., forms, buttons, input fields).\n\n**Goal:**\nThe description must be clear enough to guide a frontend developer in implementing the design.\n\n',
    "repair": "The following HTML code has some issues affecting its rendering and functionality. Please identify the problems based on the HTML code and the screenshot",
    
}

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def transform_generation(client: OpenAI, data_dir: str, output_folder: str):
    """
    Transforms HTML files in the specified directory into a JSONL format suitable for LLM training.

    Args:
        data_dir (str): Path to the directory containing HTML files.
        output_folder (str): Path to the output folder.
    """
    info = {
        "task": "generation",
        "task_type": ["single page", "no resource"],
        "description": "",
        "dst_images": [],
        "dst_code": [],
    }
    dstcode_count = 0
    images_count = 0
    folder_name = os.path.basename(data_dir.rstrip("/"))
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(data_dir):
        # consider more framework files if needed
        if filename == f"{folder_name}.html":
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
                info["dst_code"].append({
                    "language": "html",
                    "code": html_content
                })
            # 复制代码文件到输出文件夹
            dst_code_path = os.path.join(output_folder, f"dst_{dstcode_count}.html")
            with open(dst_code_path, "w", encoding="utf-8") as dst_f:
                dst_f.write(html_content)
            dstcode_count += 1
        if filename == f"{folder_name}.png":
            
            info["dst_images"].append(f"dst_{images_count}.png")
            # 复制图片到输出文件夹
            data_image_path = os.path.join(data_dir, filename)
            dst_image_path = os.path.join(output_folder, f"dst_{images_count}.png")
            with open(data_image_path, "rb") as data_f, open(dst_image_path, "wb") as dst_f:
                dst_f.write(data_f.read())
            images_count += 1
    
    # 构建消息内容
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": description_prompt_mapping["generation"]
                }
            ]
        }
    ]
    
    # 添加图片
    for image_name in info["dst_images"]:
        image_path = os.path.join(output_folder, image_name)
        base64_image = encode_image(image_path)
        message[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })
    
    # 调用 API 生成描述
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message,
            max_tokens=2000,
            temperature=0.7
        )
        
        # 提取生成的描述
        info["description"] = response.choices[0].message.content
        
    except Exception as e:
        print(f"Error calling API: {e}")
        info["description"] = ""
    
    # 写入输出文件
    info_file = os.path.join(output_folder, "info.json")
    with open(info_file, "w", encoding="utf-8") as out_f:
        json.dump(info, out_f, ensure_ascii=False, indent=4)
    
    print(f"Successfully generated description and saved to {info_file}")

def transform_edit(data_dir: str, output_folder: str):
    info = {
        "task": "edit",
        "task_type": [],
        "description": "",
        "src_images": [],
        "dst_images": [],
        "dst_code": [],
        "src_code": [],
    }
    src_code_count = 0
    dstcode_count = 0
    images_count = 0
    
    folder_name = os.path.basename(data_dir.rstrip("/"))
    os.makedirs(output_folder, exist_ok=True)
    data_json_path = os.path.join(data_dir, f"{folder_name}.json")
    data_json_info = json.load(open(data_json_path, "r", encoding="utf-8"))
    info["description"] = data_json_info['prompt']
    if "Change" in data_json_info['action_type']:
        info["task_type"].append("change style")
    if "Add" in data_json_info['action_type'] or "Delete" in data_json_info['action_type']:
        info["task_type"].append("add or del element")
    
    src_code = data_json_info['src_code']
    dst_code = data_json_info['dst_code']
    if isinstance(src_code, dict):
        for language, code in data_json_info['src_code'].items():
            info["src_code"].append({
                "language": f"{language}",
                "code": code
            })
            # 复制代码文件到输出文件夹
            src_code_path = os.path.join(output_folder, f"src_{src_code_count}.{language}")
            with open(src_code_path, "w", encoding="utf-8") as src_f:
                src_f.write(code)
            src_code_count += 1
    else:
        info["src_code"].append({
            "language": "html",
            "code": src_code
        })
        # 复制代码文件到输出文件夹
        src_code_path = os.path.join(output_folder, f"src_{src_code_count}.html")
        with open(src_code_path, "w", encoding="utf-8") as src_f:
            src_f.write(src_code)
        src_code_count += 1
    if isinstance(dst_code, dict):
        for language, code in data_json_info['dst_code'].items():
            info["dst_code"].append({
                "language": f"{language}",
                "code": code
            })
            # 复制代码文件到输出文件夹
            dst_code_path = os.path.join(output_folder, f"dst_{dstcode_count}.{language}")
            with open(dst_code_path, "w", encoding="utf-8") as dst_f:
                dst_f.write(code)
            dstcode_count += 1
    else:
        info["dst_code"].append({
            "language": "html",
            "code": dst_code
        })
        # 复制代码文件到输出文件夹
        dst_code_path = os.path.join(output_folder, f"dst_{dstcode_count}.html")
        with open(dst_code_path, "w", encoding="utf-8") as dst_f:
            dst_f.write(dst_code)
        dstcode_count += 1
            
    src_file_name = f"{data_json_info['src_id']}.png"
    dst_file_name = f"{data_json_info['dst_id']}.png"
    info["src_images"].append(f"src_0.png")
    # 复制图片到输出文件夹
    data_image_path = os.path.join(data_dir, src_file_name)
    src_image_path = os.path.join(output_folder, f"src_0.png")
    with open(data_image_path, "rb") as data_f, open(src_image_path, "wb") as src_f:
        src_f.write(data_f.read())
    info["dst_images"].append(f"dst_0.png")
    # 复制图片到输出文件夹
    data_image_path = os.path.join(data_dir, dst_file_name)
    dst_image_path = os.path.join(output_folder, f"dst_0.png")
    with open(data_image_path, "rb") as data_f, open(dst_image_path, "wb") as dst_f:
        dst_f.write(data_f.read())
    with open(os.path.join(output_folder, "info.json"), "w", encoding="utf-8") as out_f:
        json.dump(info, out_f, ensure_ascii=False, indent=4)

def transform_repair(data_dir: str, output_folder: str):
    info = {
        "task": "edit",
        "task_type": [],
        "description": "",
        "src_images": [],
        "dst_images": [],
        "dst_code": [],
        "src_code": [],
    }
    src_code_count = 0
    dstcode_count = 0
    
    folder_name = os.path.basename(data_dir.rstrip("/"))
    os.makedirs(output_folder, exist_ok=True)
    data_json_path = os.path.join(data_dir, f"{folder_name}.json")
    data_json_info = json.load(open(data_json_path, "r", encoding="utf-8"))
    repaired_json_path = os.path.join(data_dir, "repaired.json")
    repaired_json_info = json.load(open(repaired_json_path, "r", encoding="utf-8"))
    info["description"] = repaired_json_info['Reasoning']
    info["task_type"].append(data_json_info['issue'])
    
    src_code = data_json_info['code']
    dst_code = repaired_json_info['Code']
    if isinstance(src_code, dict):
        for language, code in data_json_info['src_code'].items():
            info["src_code"].append({
                "language": f"{language}",
                "code": code
            })
            # 复制代码文件到输出文件夹
            src_code_path = os.path.join(output_folder, f"src_{src_code_count}.{language}")
            with open(src_code_path, "w", encoding="utf-8") as src_f:
                src_f.write(code)
            src_code_count += 1
    else:
        info["src_code"].append({
            "language": "html",
            "code": src_code
        })
        # 复制代码文件到输出文件夹
        src_code_path = os.path.join(output_folder, f"src_{dstcode_count}.html")
        with open(src_code_path, "w", encoding="utf-8") as src_f:
            src_f.write(src_code)
        src_code_count += 1
    if isinstance(dst_code, dict):
        for language, code in data_json_info['dst_code'].items():
            info["dst_code"].append({
                "language": f"{language}",
                "code": code
            })
            # 复制代码文件到输出文件夹
            dst_code_path = os.path.join(output_folder, f"dst_{dstcode_count}.{language}")
            with open(dst_code_path, "w", encoding="utf-8") as dst_f:
                dst_f.write(code)
            dstcode_count += 1
    else:
        info["dst_code"].append({
            "language": "html",
            "code": dst_code
        })
        # 复制代码文件到输出文件夹
        dst_code_path = os.path.join(output_folder, f"dst_{dstcode_count}.html")
        with open(dst_code_path, "w", encoding="utf-8") as dst_f:
            dst_f.write(dst_code)
        dstcode_count += 1
            
    src_file_name = f"{folder_name}.png"
    dst_file_name = "repaired.png"
    info["src_images"].append(f"src_0.png")
    # 复制图片到输出文件夹
    data_image_path = os.path.join(data_dir, src_file_name)
    src_image_path = os.path.join(output_folder, f"src_0.png")
    with open(data_image_path, "rb") as data_f, open(src_image_path, "wb") as src_f:
        src_f.write(data_f.read())
    info["dst_images"].append(f"dst_0.png")
    # 复制图片到输出文件夹
    data_image_path = os.path.join(data_dir, dst_file_name)
    dst_image_path = os.path.join(output_folder, f"dst_0.png")
    with open(data_image_path, "rb") as data_f, open(dst_image_path, "wb") as dst_f:
        dst_f.write(data_f.read())
    with open(os.path.join(output_folder, "info.json"), "w", encoding="utf-8") as out_f:
        json.dump(info, out_f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    config = OmegaConf.load("config/api.yaml")
    client = OpenAI(
        base_url=config.api.base_url,
        api_key=config.api.api_key,
    )
    data_dir = "/Users/pedestrian/Desktop/Webcode Dataset/DesignBench/data/generation/react/1"  # 修改为你的HTML文件目录
    output_folder = "data_demo/generation/designbench_demo_2"  # 输出文件路径
    transform_generation(client, data_dir, output_folder)
    
    # data_dir = "/Users/pedestrian/Desktop/Webcode Dataset/DesignBench/data/edit/angular/30"  # 修改为你的HTML文件目录
    # output_folder = "data_demo/edit/designbench_demo_2"  # 输出文件路径
    # transform_edit(data_dir, output_folder)
    
    # data_dir = "/Users/pedestrian/Desktop/Webcode Dataset/DesignBench/data/repair/vanilla/25"
    # output_folder = "data_demo/repair/designbench_demo_2"  # 输出文件路径
    # transform_repair(data_dir, output_folder)