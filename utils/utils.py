import os
import base64
import cairosvg
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI
import traceback
import shutil
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from utils.config import CODE_EXTENSIONS

def encode_image(image_path):
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext == ".svg":
        # 将SVG转换为PNG字节流，不保存到磁盘
        png_content = cairosvg.svg2png(url=image_path)
        return base64.b64encode(png_content).decode("utf-8")

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def chat_with_retry(
    client: OpenAI,
    messages: list,
    model: str,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    max_retries: int = 3,
    retry_delay: int = 2,
    **kwargs,
) -> Optional[str]:
    """
    带重试机制的聊天函数

    Args:
        client: OpenAI客户端实例
        messages: 消息列表
        model: 模型名称
        max_tokens: 最大token数
        temperature: 温度参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟(秒)
        **kwargs: 其他传递给API的参数

    Returns:
        成功时返回响应内容,失败时返回None
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            print(f"Full traceback: {traceback.format_exc()}")

            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)  # 指数退避
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"达到最大重试次数,调用失败")
                return None

    return None

def save_screenshots(directory: str) -> List[str]:
    """
    遍历目录下的所有HTML文件并保存截图。
    
    Args:
        directory: 要遍历的目录路径
        
    Returns:
        截图文件相对路径列表
    """
    screenshot_paths = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return screenshot_paths
    
    # 查找所有HTML文件
    html_files = list(dir_path.glob("**/*.html"))
    
    if not html_files:
        return screenshot_paths
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        for html_file in html_files:
            try:
                page = browser.new_page(
                    viewport={"width": 1280, "height": 720},
                    device_scale_factor=2,
                )
                
                url = html_file.as_uri()
                page.goto(url, wait_until="networkidle")
                page.wait_for_timeout(200)
                
                # 生成截图文件名（与html文件同名，扩展名改为.png）
                rel_path = html_file.relative_to(dir_path)
                screenshot_name = f"screenshot_{rel_path.stem}.png"
                screenshot_path = dir_path / screenshot_name
                
                page.screenshot(
                    path=str(screenshot_path),
                    full_page=True,
                    animations="disabled",
                )
                
                screenshot_paths.append(screenshot_name)
                page.close()
                
            except Exception as e:
                print(f"截图失败 {html_file}: {e}")
                continue
        
        browser.close()
    
    return screenshot_paths


def save_task(
    task: Dict,
    output_base_dir: str,
    task_id: str,
    source_generation_dir: str = None,
) -> None:
    """
    Save a task to the filesystem with proper directory structure.

    This is a generic function that can be used for both edit and repair tasks.

    Structure:
    output_base_dir/
    └── {task_id}/
        ├── src/
        │   ├── index.html
        │   └── resources/
        │       └── ...
        ├── dst/
        │   ├── index.html
        │   └── resources/
        │       └── ...
        ├── info.json
        └── llm_log.json

    Args:
        task: The task dictionary containing src_code and dst_code
        output_base_dir: Base directory for output
        task_id: Unique identifier for this task
        source_generation_dir: Path to the original generation folder (for copying resources)
    """


    task_dir = os.path.join(output_base_dir, task_id)
    src_dir = os.path.join(task_dir, "src")
    dst_dir = os.path.join(task_dir, "dst")

    # Create directories
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)

    # Copy non-code resources from generation dst folder if source_generation_dir is provided
    if source_generation_dir:
        gen_dst_dir = os.path.join(source_generation_dir, "dst")
        if os.path.exists(gen_dst_dir):
            for item in os.listdir(gen_dst_dir):
                item_path = os.path.join(gen_dst_dir, item)
                # Skip screenshot files
                if item.startswith("screenshot"):
                    continue
                # Handle directories (like resources/)
                if os.path.isdir(item_path):
                    # Copy to both src and dst
                    src_resource_dir = os.path.join(src_dir, item)
                    dst_resource_dir = os.path.join(dst_dir, item)

                    # Copy directory, excluding code files
                    for root, dirs, files in os.walk(item_path):
                        rel_path = os.path.relpath(root, item_path)
                        for file in files:
                            file_ext = os.path.splitext(file)[1].lower()
                            # Only copy non-code files (images, fonts, etc.)
                            if file_ext not in CODE_EXTENSIONS:
                                src_file = os.path.join(root, file)

                                # Copy to src directory
                                dst_src_path = os.path.join(
                                    src_resource_dir, rel_path, file
                                )
                                os.makedirs(
                                    os.path.dirname(dst_src_path), exist_ok=True
                                )
                                shutil.copy2(src_file, dst_src_path)

                                # Copy to dst directory
                                dst_dst_path = os.path.join(
                                    dst_resource_dir, rel_path, file
                                )
                                os.makedirs(
                                    os.path.dirname(dst_dst_path), exist_ok=True
                                )
                                shutil.copy2(src_file, dst_dst_path)
                # Handle single files at root level (non-code, non-screenshot)
                elif os.path.isfile(item_path):
                    file_ext = os.path.splitext(item)[1].lower()
                    if file_ext not in CODE_EXTENSIONS:
                        shutil.copy2(item_path, os.path.join(src_dir, item))
                        shutil.copy2(item_path, os.path.join(dst_dir, item))

    # Save src code files
    for file_info in task["src_code"]:
        file_path = os.path.join(src_dir, file_info["path"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_info["code"])

    # Save dst code files
    for file_info in task["dst_code"]:
        file_path = os.path.join(dst_dir, file_info["path"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_info["code"])

    # Save screenshots
    src_screenshots = save_screenshots(src_dir)
    dst_screenshots = save_screenshots(dst_dir)
    
    # Prepare info.json
    info = {
        "task": task["task"],
        "task_type": task["task_type"],
        "description": task["description"],
        "src_code": task["src_code"],
        "dst_code": task["dst_code"],
        "src_screenshot": src_screenshots,
        "dst_screenshot": dst_screenshots,
        "label_modified_files": task.get("label_modified_files", []),
        "resources": task.get("resources", []),
    }

    llm_systhetic_log = {
        "llm_raw_response": task.get("llm_raw_response", ""),
        "llm_metadata": task.get("llm_metadata", {}),
        "synthetic_modified_files": task.get("synthetic_modified_files", []),
    }

    # Save info.json
    info_path = os.path.join(task_dir, "info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # Save llm_log.json
    llm_log_path = os.path.join(task_dir, "llm_systhetic_log.json")
    with open(llm_log_path, "w", encoding="utf-8") as f:
        json.dump(llm_systhetic_log, f, indent=2, ensure_ascii=False)

    print(f"Saved task to {task_dir}")
