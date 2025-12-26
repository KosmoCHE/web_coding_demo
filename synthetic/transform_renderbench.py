import base64
from openai import OpenAI
import os
import json
import shutil
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import encode_image, chat_with_retry
import traceback
from utils.config import *
# 统一使用的模型
MODEL_NAME = "gpt-4o-mini"



description_prompt_mapping = {
    "generation": '**Role:**\nYou are a senior Frontend Product Manager and UI Designer. Please analyze the provided webpage screenshot to generate a concise requirement description for frontend development.\n\n**Task:**\nAnalyze the provided screenshot and summarize the webpage\'s core visual characteristics and key interactions using **very concise natural language (under 150 words)**.\n\n**Strict Rules:**\n- **Zero Code Principle:** Do NOT use any code snippets, HTML tags, or technical CSS parameters.\n- **Visuals Only:** Describe only what a user can "see" and "feel." For example, instead of saying "set border-radius to 50%," say "the avatar is displayed as a perfect circle."\n- **Macro Perspective:** Focus on the overall layout (Header/Body/Footer), the primary color palette/mood, the visual style, and the core functional components (e.g., forms, buttons, input fields).\n\n**Goal:**\nThe description must be clear enough to guide a frontend developer in implementing the design.\n\n',
}


def get_image_description(client: OpenAI, file_path: str, filename: str):
    """获取单个图片的描述"""
    base64_image = encode_image(file_path)
    
    # 根据文件扩展名确定MIME类型
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == ".svg":
        mime_type = "image/png"  # SVG已转换为PNG
    elif file_ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif file_ext == ".png":
        mime_type = "image/png"
    else:
        mime_type = "image/png"  # 默认使用PNG
    
    image_description_message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please describe this image briefly in 1-2 sentences, focusing on its visual content and purpose.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    },
                },
            ],
        }
    ]

    description = chat_with_retry(
        client=client,
        messages=image_description_message,
        model=MODEL_NAME,
        max_tokens=200,
        temperature=0.7,
        max_retries=6
    )

    return {
        "type": "image",
        "path": f"resources/{filename}",
        "description": description if description else "",
    }


def transform_generation(
    client: OpenAI, dataset_base_path: str, web_name: str, output_folder: str
):
    """
    Transforms HTML files in the specified directory into a JSONL format suitable for LLM training.

    Args:
        web_name (str): Name of the webpage.
        output_folder (str): Path to the output folder.
    """
    info = {
        "task": "generation",
        "task_type": ["single page", "with resource"],
        "description": "",
        "dst_screenshot": [],
        "dst_code": [],
        "resources": [],
    }
    dstcode_count = 0
    screenshot_count = 0
    webpage_folder = dataset_base_path + "/train_webpages/" + web_name
    screenshot_name = (
        dataset_base_path + "/train_screenshots/" + web_name + ".jpg"
    )
    os.makedirs(output_folder, exist_ok=True)
    # mkdir "/dst" folder
    dst_repo_dir = os.path.join(output_folder, "dst")
    os.makedirs(dst_repo_dir, exist_ok=True)

    resources_data_dir = os.path.join(webpage_folder, "resources")
    resources_dst = os.path.join(dst_repo_dir, "resources")
    if os.path.exists(resources_data_dir):
        if os.path.exists(resources_dst):
            shutil.rmtree(resources_dst)  # 如果目标文件夹已存在,先删除
        shutil.copytree(resources_data_dir, resources_dst)

        # 处理resources中的非代码数据
        # 收集需要处理的图片和其他文件
        image_files = []
        other_files = []

        for filename in os.listdir(resources_data_dir):
            file_path = os.path.join(resources_data_dir, filename)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(filename)[1].lower()

                # 跳过代码文件
                if file_ext in CODE_EXTENSIONS:
                    continue

                # 分类图片文件和其他文件
                if file_ext in IMAGE_EXTENSIONS:
                    image_files.append((file_path, filename))
                else:
                    other_files.append(filename)

        # 多线程处理图片描述
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_image = {
                executor.submit(
                    get_image_description, client, file_path, filename
                ): filename
                for file_path, filename in image_files
            }

            for future in as_completed(future_to_image):
                result = future.result()
                info["resources"].append(result)

        # 处理其他非代码文件
        for filename in other_files:
            info["resources"].append(
                {
                    "type": "other",
                    "path": f"resources/{filename}",
                    "description": "",
                }
            )

    # 处理截图
    with open(screenshot_name, "rb") as img_f:
        img_data = img_f.read()
        screenshot_output_path = os.path.join(
            dst_repo_dir, f"screenshot_{screenshot_count}.jpg"
        )
        with open(screenshot_output_path, "wb") as out_img_f:
            out_img_f.write(img_data)
        info["dst_screenshot"].append(f"screenshot_{screenshot_count}.jpg")
        screenshot_count += 1
    # 处理index.html文件
    with open(
        os.path.join(webpage_folder, "index.html"), "r", encoding="utf-8"
    ) as f:
        html_content = f.read()
        info["dst_code"].append(
            {
                "directory": "index.html",
                "code": html_content,
            }
        )
        # 复制代码文件到输出文件夹
        dst_code_path = os.path.join(dst_repo_dir, f"index.html")
        with open(dst_code_path, "w", encoding="utf-8") as dst_f:
            dst_f.write(html_content)
        dstcode_count += 1

    for filename in os.listdir(resources_data_dir):
        file_path = os.path.join(webpage_folder, "resources", filename)
        file_ext = os.path.splitext(filename)[1].lower()

        # 只要是定义的代码种类就记录
        if file_ext in CODE_EXTENSIONS:
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()
                info["dst_code"].append(
                    {
                        "path": f"resources/{filename}",
                        
                        "code": code_content,
                    }
                )
                dstcode_count += 1

    # 构建消息内容
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": description_prompt_mapping["generation"],
                }
            ],
        }
    ]

    # 添加图片
    for image_name in info["dst_screenshot"]:
        image_path = os.path.join(dst_repo_dir, image_name)
        base64_image = encode_image(image_path)
        
        # 根据文件扩展名确定MIME类型
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif file_ext == ".png":
            mime_type = "image/png"
        elif file_ext == ".svg":
            mime_type = "image/png"  # SVG已转换为PNG
        else:
            mime_type = "image/png"  # 默认使用PNG
        
        message[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            }
        )

    # 调用 API 生成描述
    description = chat_with_retry(
        client=client,
        messages=message,
        model=MODEL_NAME,
        max_tokens=4096,
        temperature=0.7,
        max_retries=6
    )

    # 提取生成的描述
    info["description"] = description if description else ""

    # 写入输出文件
    info_file = os.path.join(output_folder, "info.json")
    with open(info_file, "w", encoding="utf-8") as out_f:
        json.dump(info, out_f, ensure_ascii=False, indent=4)

    print(f"Successfully generated description and saved to {info_file}")


if __name__ == "__main__":
    config = OmegaConf.load("config/api.yaml")
    client = OpenAI(
        base_url=config.api.base_url,
        api_key=config.api.api_key,
    )

    dataset_base_path = "/Users/pedestrian/Desktop/Webcode Dataset/small_webrender_bench"  # 数据集文件夹路径
    
    # # test single web
    # web_name = "2919922_www.tiropartners.com"  # 网页名称
    # output_folder = (
    #     f"data_demo_renderbench/test_generation/{web_name}"  # 输出文件路径
    # )
    # transform_generation(client, dataset_base_path, web_name, output_folder)

    dataset_info = dataset_base_path + "/demo_web_render_train.jsonl"
    with open(dataset_info, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            web_name = data["name"]
            output_folder = f"data/data_demo_renderbench/generation/{web_name}"
            transform_generation(
                client, dataset_base_path, web_name, output_folder
            )
