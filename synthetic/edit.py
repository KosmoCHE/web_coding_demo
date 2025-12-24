from openai import OpenAI
import os
import json
from typing import List, Dict, Any
from omegaconf import OmegaConf
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from synthesizer import BaseSynthesizer

# Task Categories
FORWARD_TASKS = [
    # "Responsive Adaptation",
    # "Page Addition",
    "Function Modification",
    "Element Modification",
    "Add Element",
    "Style Modification",
    "Remove Element"
]

# 每种任务类型的详细说明
TASK_DESCRIPTIONS = {
    "Function Modification": """Modify an existing JavaScript function or interaction behavior.
    For example, change a click handler to show a different message, modify form validation logic, 
    or update an animation timing.""",
    
    "Element Modification": """Modify an existing HTML element's content or attributes.
    For example, change button text, update link href, modify image src, or change heading content.""",
    
    "Add Element": """Add a new HTML element to the page.
    For example, add a new button, create a new section, insert a banner, or add a footer element.""",
    
    "Style Modification": """Modify CSS styles of existing elements.
    For example, change colors, adjust spacing, modify fonts, update borders, or change layout properties.""",
    
    "Remove Element": """Remove an existing HTML element from the page.
    For example, delete a popup, remove a banner, eliminate a button, or take out a section.""",
}


class EditTaskSynthesizer(BaseSynthesizer):
    
    def generate_forward_task(self, generation_data: Dict, task_type: str) -> Dict:
        """
        策略1：前向演化
        现有代码(Generation) -> 源代码
        LLM修改它 -> 目标代码
        使用 search/replace 模式
        """
        src_code_context = self.format_code_context(generation_data["dst_code"])
        task_description = TASK_DESCRIPTIONS.get(task_type, "")

        prompt = f"""You are an expert web developer. I have a codebase for a webpage.
I want to generate a dataset for web editing tasks.

Current Task Type: {task_type}

Task Instructions:
{task_description}

Please analyze the provided code and propose a specific, reasonable modification task that fits the '{task_type}' category.
Then, implement the modification using search/replace blocks.

Return XML format with the following structure:
<description>A clear, one-sentence instruction for the task (e.g., 'Change the button color to red' or 'Add a navigation bar').</description>
<search_replace path="path/to/file">
<search>
exact text to find in the original file
</search>
<replace>
replacement text with the modification applied
</replace>
</search_replace>

You can include multiple <search_replace> blocks if you need to modify multiple locations.

Important:
- The <search> block must contain the EXACT text from the original file (including whitespace and indentation).
- The <replace> block contains the modified code.
- Do not include 'src' or 'dst' labels in the description, just the task instruction.
- Make sure the modifications are obvious enough in visual appearance to be noticed in screenshots.
- Keep the search blocks as small as possible while still being unique in the file.

Here is the source code:
{src_code_context}"""

        try:
            response = self._create_chat_completion_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates web development datasets in XML format using search/replace blocks.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            result = response
            src_code = generation_data["dst_code"]
            # 使用 search/replace 应用修改
            dst_code = self.apply_search_replace(
                src_code, 
                result.get("modified_files", [])
            )

            return {
                "task": "edit",
                "task_type": [task_type],
                "description": result["description"],
                "src_code": src_code,
                "dst_code": dst_code,
                "src_screenshot": generation_data.get("dst_screenshot", []),
                "dst_screenshot": [],
                "modified_files": result.get("modified_files", []),  # 记录 search/replace 块
                "llm_raw_response": result.get("raw_response"),
                "llm_metadata": result.get("llm_metadata"),
            }
        except Exception as e:
            print(f"Error generating forward task ({task_type}): {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_single_generation_entry(
        self,
        generation_entry: Dict,
        output_dir: str = None,
        folder_name: str = None,
        source_generation_dir: str = None,
        forward_tasks: List[str] = None,
        reverse_tasks: List[str] = None,
    ) -> List[Dict]:
        """
        处理单个 generation entry，为每种任务类型生成编辑任务
        
        Args:
            generation_entry: 原始 generation 数据
            output_dir: 输出目录
            folder_name: 文件夹名称
            source_generation_dir: 源 generation 目录
            forward_tasks: 要生成的前向任务类型列表，默认为所有类型
            reverse_tasks: 要生成的反向任务类型列表，默认为所有类型
        """
        generated_tasks = []
        task_index = 0

        # 如果没有指定任务类型，使用默认类型
        if forward_tasks is None:
            forward_tasks = FORWARD_TASKS

        # Forward Tasks
        for fwd_task_type in forward_tasks:
            print(f"Generating Forward Task: {fwd_task_type}")
            task = self.generate_forward_task(generation_entry, fwd_task_type)
            if task:
                generated_tasks.append(task)
                # 立即保存
                if output_dir and folder_name:
                    task_type_suffix = (
                        task["task_type"][0].lower().replace(" ", "_")
                    )
                    task_id = f"{folder_name}_{task_type_suffix}_{task_index}"
                    save_edit_task(
                        task,
                        output_dir,
                        task_id,
                        source_generation_dir=source_generation_dir,
                    )
                task_index += 1

        return generated_tasks


def save_edit_task(
    task: Dict,
    output_base_dir: str,
    task_id: str,
    source_generation_dir: str = None,
) -> None:
    """
    Save an edit task to the filesystem with proper directory structure.

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

    # Get code file extensions to exclude from resource copying
    code_extensions = {
        ".html",
        ".css",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".vue",
        ".svelte",
    }

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
                            if file_ext not in code_extensions:
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
                    if file_ext not in code_extensions:
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

    # Prepare info.json - 现在记录 modified_files (search/replace 块)
    info = {
        "task": task["task"],
        "task_type": task["task_type"],
        "description": task["description"],
        "src_code": task["src_code"],
        "dst_code": task["dst_code"],
        "src_screenshot": task.get("src_screenshot", []),
        "dst_screenshot": task.get("dst_screenshot", []),
        "modified_files": task.get("modified_files", []),
    }
    llm_log = {
        "llm_raw_response": task.get("llm_raw_response", ""),
        "llm_metadata": task.get("llm_metadata", {}),
    }

    # Save info.json
    info_path = os.path.join(task_dir, "info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    # Save llm_log.json
    llm_log_path = os.path.join(task_dir, "llm_log.json")
    with open(llm_log_path, "w", encoding="utf-8") as f:
        json.dump(llm_log, f, indent=2, ensure_ascii=False)
    
    print(f"Saved edit task to {task_dir}")


def process_single_info_json(args):
    """
    处理单个 info.json 文件的辅助函数

    Args:
        args: 包含参数的元组

    Returns:
        处理的任务数量
    """
    full_path, original_folder_name, root, output_dir, synthesizer = args

    print(f"Processing {full_path}...")
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            gen_data = json.load(f)

        new_tasks = synthesizer.process_single_generation_entry(
            gen_data,
            output_dir=output_dir,
            folder_name=original_folder_name,
            source_generation_dir=root,
        )
        print(f"✓ Completed {full_path}: {len(new_tasks)} tasks")
        return len(new_tasks)
    except Exception as e:
        print(f"✗ Error processing {full_path}: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main(max_workers=4):
    """
    主函数 - 多线程版本

    Args:
        max_workers: 最大线程数,默认为4
    """
    # Configuration
    input_dir = (
        "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/generation"
    )
    output_dir = "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/edit"
    config = OmegaConf.load("config/api.yaml")
    api_key = config.api.api_key
    base_url = config.api.base_url
    model = "claude-3-5-sonnet-coder"

    synthesizer = EditTaskSynthesizer(api_key, base_url, model, max_tokens=32*1024)

    # 收集所有需要处理的文件
    tasks_to_process = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == "info.json":
                full_path = os.path.join(root, file)
                original_folder_name = os.path.basename(root)
                tasks_to_process.append(
                    (
                        full_path,
                        original_folder_name,
                        root,
                        output_dir,
                        synthesizer,
                    )
                )

    print(f"Found {len(tasks_to_process)} generation folders to process")
    print(f"Using {max_workers} worker threads")

    task_counter = 0

    # 使用线程池处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_path = {
            executor.submit(process_single_info_json, task_args): task_args[0]
            for task_args in tasks_to_process
        }

        # 处理完成的任务
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                task_count = future.result()
                task_counter += task_count
            except Exception as e:
                print(f"Exception for {path}: {e}")

    print(f"\n{'='*60}")
    print(f"✓ All tasks completed!")
    print(f"Generated {task_counter} edit tasks in {output_dir}")
    print(f"{'='*60}")


def test_single_generation(
    generation_folder: str, 
    output_dir: str = None,
    forward_tasks: List[str] = None,
    reverse_tasks: List[str] = None,
) -> List[Dict]:
    """
    测试函数:处理指定的单个 generation 文件夹

    Args:
        generation_folder: generation 文件夹的路径
        output_dir: 输出目录,如果为 None 则不保存文件,只返回结果
        forward_tasks: 要生成的前向任务类型列表，默认为所有类型
        reverse_tasks: 要生成的反向任务类型列表，默认为所有类型

    Returns:
        生成的 edit task 列表

    Example:
        # 只生成不保存
        tasks = test_single_generation("/path/to/generation/folder")

        # 生成并保存，只测试部分任务类型
        tasks = test_single_generation(
            "/path/to/generation/folder", 
            "/path/to/output",
            forward_tasks=["Style Modification"],
            reverse_tasks=[]
        )
    """
    config = OmegaConf.load("config/api.yaml")
    api_key = config.api.api_key
    base_url = config.api.base_url
    model = "claude-3-5-sonnet-coder"
    synthesizer = EditTaskSynthesizer(api_key, base_url, model, max_tokens=16*1024)

    info_path = os.path.join(generation_folder, "info.json")

    if not os.path.exists(info_path):
        print(f"Error: info.json not found in {generation_folder}")
        return []

    original_folder_name = os.path.basename(generation_folder)
    print(f"Processing {info_path}...")

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            gen_data = json.load(f)

        new_tasks = synthesizer.process_single_generation_entry(
            gen_data,
            output_dir=output_dir,
            folder_name=original_folder_name,
            source_generation_dir=generation_folder,
            forward_tasks=forward_tasks,
            reverse_tasks=reverse_tasks,
        )

        print(f"Generated {len(new_tasks)} edit tasks")

        # 打印任务摘要
        for i, task in enumerate(new_tasks):
            print(f"\n--- Task {i+1} ---")
            print(f"Type: {task['task_type']}")
            print(f"Description: {task['description']}")
            print(
                f"Modified files: {[f['path'] for f in task.get('modified_files', [])]}"
            )

        return new_tasks

    except Exception as e:
        print(f"Error processing {info_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_single_task_type(
    generation_folder: str,
    task_type: str,
    output_dir: str = None,
) -> List[Dict]:
    """
    测试单个任务类型的生成

    Args:
        generation_folder: generation 文件夹的路径
        task_type: 要测试的任务类型
        output_dir: 输出目录

    Returns:
        生成的 edit task 列表
    """
    if task_type in FORWARD_TASKS:
        return test_single_generation(
            generation_folder, 
            output_dir, 
            forward_tasks=[task_type],
            reverse_tasks=[]
        )
    elif task_type in REVERSE_TASKS:
        return test_single_generation(
            generation_folder, 
            output_dir, 
            forward_tasks=[],
            reverse_tasks=[task_type]
        )
    else:
        print(f"Error: Invalid task type '{task_type}'")
        print(f"Valid forward types: {FORWARD_TASKS}")
        print(f"Valid reverse types: {REVERSE_TASKS}")
        return []


if __name__ == "__main__":
    # 可以调整线程数,建议根据 API 限流情况设置
    # main(max_workers=5)

    # # 或者测试单个文件夹
    # tasks = test_single_generation(
    #     "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/generation/1009769_www.kccworld.co.kr_english_",
    #     "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/edit_test_sr",
    #     forward_tasks=["Style Modification"],
    #     reverse_tasks=["Remove Element"]
    # )
    
    # 或者测试单个任务类型
    tasks = test_single_task_type(
        "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/generation/1009769_www.kccworld.co.kr_english_",
        "Style Modification",
        "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/edit_test_sr"
    )
