import base64
from openai import OpenAI
import os
import json
import shutil
from omegaconf import OmegaConf
description_prompt_mapping = {
    "generation": '**Role:**\nYou are a senior Frontend Product Manager and UI Designer. Please analyze the provided webpage screenshot to generate a concise requirement description for frontend development.\n\n**Task:**\nAnalyze the provided screenshot and summarize the webpage\'s core visual characteristics and key interactions using **very concise natural language (under 150 words)**.\n\n**Strict Rules:**\n- **Zero Code Principle:** Do NOT use any code snippets, HTML tags, or technical CSS parameters.\n- **Visuals Only:** Describe only what a user can "see" and "feel." For example, instead of saying "set border-radius to 50%," say "the avatar is displayed as a perfect circle."\n- **Macro Perspective:** Focus on the overall layout (Header/Body/Footer), the primary color palette/mood, the visual style, and the core functional components (e.g., forms, buttons, input fields).\n\n**Goal:**\nThe description must be clear enough to guide a frontend developer in implementing the design.\n\n',
}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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
                "language": "html",
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

        # 识别不同类型的代码文件
        if filename.endswith(".css"):
            with open(file_path, "r", encoding="utf-8") as f:
                css_content = f.read()
                info["dst_code"].append(
                    {
                        "language": "css",
                        "directory": f"resources/{filename}",
                        "code": css_content,
                    }
                )
                dstcode_count += 1

        elif filename.endswith(".js"):
            with open(file_path, "r", encoding="utf-8") as f:
                js_content = f.read()
                info["dst_code"].append(
                    {
                        "language": "javascript",
                        "directory": f"resources/{filename}",
                        "code": js_content,
                    }
                )
                dstcode_count += 1

        elif filename.endswith(".html"):
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
                info["dst_code"].append(
                    {
                        "language": "html",
                        "directory": f"resources/{filename}",
                        "code": html_content,
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
        message[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    # 调用 API 生成描述
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=message,
            max_tokens=2000,
            temperature=0.7,
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


if __name__ == "__main__":
    config = OmegaConf.load("config/api.yaml")
    client = OpenAI(
        base_url=config.api.base_url,
        api_key=config.api.api_key,
    )

    dataset_base_path = "/Users/pedestrian/Desktop/Webcode Dataset/small_webrender_bench"  # 数据集文件夹路径
    # web_name = "2919922_www.tiropartners.com"  # 网页名称
    # output_folder = (
    #     f"data_demo_renderbench/generation/{web_name}"  # 输出文件路径
    # )
    # transform_generation(client, dataset_base_path, web_name, output_folder)
    dataset_info = dataset_base_path + "/demo_web_render_train.jsonl"
    with open(dataset_info, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            web_name = data["name"]
            output_folder = f"data_demo_renderbench/generation/{web_name}"
            transform_generation(
                client, dataset_base_path, web_name, output_folder
            )