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
]

REVERSE_TASKS = ["Remove Element"]


class EditTaskSynthesizer(BaseSynthesizer):
    
    def generate_forward_task(self, generation_data: Dict, task_type: str) -> Dict:
        """
        策略1：前向演化
        现有代码(Generation) -> 源代码
        LLM修改它 -> 目标代码
        """
        src_code_context = self.format_code_context(generation_data["dst_code"])

        prompt =f"""You are an expert web developer. I have a codebase for a webpage.
I want to generate a dataset for web editing tasks.
Current Task Type: {task_type}

Please analyze the provided code and propose a specific, reasonable modification task that fits the '{task_type}' category.
Then, implement the modification.

Return XML format with the following structure:
<description>A clear, one-sentence instruction for the task (e.g., 'Change the button color to red' or 'Add a navigation bar').</description>
<file path="path/to/file">
The full content of the modified file
</file>
<file path="path/to/another/file">
The full content of another modified file
</file>

Only include files that were actually modified.
Do not include 'src' or 'dst' labels in the description, just the task instruction.
Make sure the modifications is obvious enough in visual appearance to be noticed in screenshots.
Here is the source code:
{src_code_context}"""

        try:
            response = self._create_chat_completion_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates web development datasets in XML format.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            result = response
            src_code = generation_data["dst_code"]
            dst_code = self.merge_code_with_modifications(
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
                "modified_files": result.get("modified_files", []),
                "llm_raw_response": result.get("raw_response"),
                "llm_metadata": result.get("llm_metadata"),
            }
        except Exception as e:
            print(f"Error generating forward task: {e}")
            return None
    
    def generate_reverse_task(self, generation_data: Dict, task_type: str) -> Dict:
        """
        策略2：缺陷注入/反向简化
        现有代码(Generation) -> 目标代码（干净的结果）
        LLM添加缺陷/元素 -> 源代码（起点）
        """
        dst_code_context = self.format_code_context(generation_data["dst_code"])

        prompt = f"""You are an expert web developer. I have a clean, high-quality codebase for a webpage.
I want to generate a dataset for '{task_type}' tasks (specifically removing elements).

Please analyze the provided code and inject a "defect" or an extra element that a user would want to remove.
For example, add a "Subscribe to Newsletter" popup that blocks content, or a redundant banner, or a deprecated button.

Return XML format with the following structure:
<description>A clear, one-sentence instruction for the removal task (e.g., 'Remove the newsletter popup' or 'Delete the promotional banner').</description>
<file path="path/to/file">
The full content of the file with the added element/defect
</file>
<file path="path/to/another/file">
The full content of another modified file
</file>

Only include files that were modified, you must make actual code changes to reflect the added defect/element.
Make sure the modifications is obvious enough in visual appearance to be noticed in screenshots.
Here is the clean code (which will be the Goal/Dst state):
{dst_code_context}"""

        try:
            response = self._create_chat_completion_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates web development datasets in XML format.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            result = response
            dst_code = generation_data["dst_code"]
            src_code = self.merge_code_with_modifications(
                dst_code,
                result.get("modified_files", [])
            )

            return {
                "task": "edit",
                "task_type": [task_type],
                "description": result["description"],
                "src_code": src_code,
                "dst_code": dst_code,
                "src_screenshot": [],
                "dst_screenshot": generation_data.get("dst_screenshot", []),
                "modified_files": result.get("modified_files", []),
                "llm_raw_response": result.get("raw_response"),
                "llm_metadata": result.get("llm_metadata"),
            }
        except Exception as e:
            print(f"Error generating reverse task: {e}")
            return None

    def process_single_generation_entry(
        self,
        generation_entry: Dict,
        output_dir: str = None,
        folder_name: str = None,
        source_generation_dir: str = None,
    ) -> List[Dict]:
        generated_tasks = []
        task_index = 0

        # Forward Tasks
        for fwd_task_type in FORWARD_TASKS:
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

        # Reverse Tasks
        for rev_task_type in REVERSE_TASKS:
            print(f"Generating Reverse Task: {rev_task_type}")
            task = self.generate_reverse_task(generation_entry, rev_task_type)
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
        └── info.json

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
        file_path = os.path.join(src_dir, file_info["directory"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_info["code"])

    # Save dst code files
    for file_info in task["dst_code"]:
        file_path = os.path.join(dst_dir, file_info["directory"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_info["code"])

    # Prepare info.json (include full code content)
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
    print(f"Saved task to {task_dir}")


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

    # 注意: 每个线程需要自己的 synthesizer 实例以避免共享状态问题
    # 但 OpenAI client 通常是线程安全的,所以我们可以共享一个 synthesizer
    synthesizer = EditTaskSynthesizer(api_key, base_url, model)

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
    generation_folder: str, output_dir: str = None
) -> List[Dict]:
    """
    测试函数:处理指定的单个 generation 文件夹

    Args:
        generation_folder: generation 文件夹的路径,例如 "/path/to/generation/1009769_www.kccworld.co.kr_english_"
        output_dir: 输出目录,如果为 None 则不保存文件,只返回结果

    Returns:
        生成的 edit task 列表

    Example:
        # 只生成不保存
        tasks = test_single_generation("/path/to/generation/folder")

        # 生成并保存
        tasks = test_single_generation("/path/to/generation/folder", "/path/to/output")
    """
    config = OmegaConf.load("config/api.yaml")
    api_key = config.api.api_key
    base_url = config.api.base_url
    model = "claude-3-5-sonnet-coder"  # Get model from config or use default
    synthesizer = EditTaskSynthesizer(api_key, base_url, model)

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
        )

        print(f"Generated {len(new_tasks)} edit tasks")

        # 打印任务摘要
        for i, task in enumerate(new_tasks):
            print(f"\n--- Task {i+1} ---")
            print(f"Type: {task['task_type']}")
            print(f"Description: {task['description']}")
            print(
                f"Modified files: {[f['directory'] for f in task.get('modified_files', [])]}"
            )

        return new_tasks

    except Exception as e:
        print(f"Error processing {info_path}: {e}")
        import traceback

        traceback.print_exc()
        return []


def process_single_info_json(args):
    """
    处理单个 info.json 文件的辅助函数

    Args:
        args: 包含 (full_path, original_folder_name, root, output_dir, synthesizer) 的元组

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


if __name__ == "__main__":
    # 可以调整线程数,建议根据 API 限流情况设置
    main(max_workers=5)

    # # 或者测试单个文件夹
    # tasks = test_single_generation(
    #     "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/generation/1009769_www.kccworld.co.kr_english_",
    #     "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/edit_test_xml_new"
    # )
