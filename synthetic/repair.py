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
import random

# 破坏类型分类
DEFECT_TYPES = [
    "Occlusion",           # 遮挡
    "Crowding",            # 拥挤
    "Text Overlap",        # 文本重叠
    "Alignment",           # 对齐问题
    "Color Contrast",      # 颜色与对比度
    "Overflow",            # 溢出
    "Sizing Proportion",   # 尺寸/比例失衡
    "Responsiveness",      # 响应式缺陷
    "Loss of Interactivity",  # 交互性丧失
    "Semantic Error",      # 语义错误
    "Nesting Error",       # 嵌套错误
    "Missing Attributes",  # 属性缺失
]

# 每种破坏类型的详细说明
DEFECT_DESCRIPTIONS = {
    "Occlusion": """Increase the z-index of element A so that it covers element B. 
    For example, make a modal overlay cover important content, or make a fixed header cover interactive elements.""",
    
    "Crowding": """Remove margin or padding between elements A and B, or shrink their parent container size.
    For example, remove spacing between navigation items, or collapse the gap between form fields.""",
    
    "Text Overlap": """Reduce the width or line-height of a text container, or position two text containers at the same location.
    For example, make text overflow its container and overlap with adjacent elements.""",
    
    "Alignment": """Adjust the left/top properties of element A so it's not aligned with the grid or sibling element B.
    For example, misalign navigation items, or offset a button from its expected position.""",
    
    "Color Contrast": """Set text color to a value similar to the background color (e.g., light gray text on white background).
    For example, make body text nearly invisible, or reduce contrast of important labels.""",
    
    "Overflow": """Add excessive content to a fixed height/width container and set overflow: visible or remove overflow handling.
    For example, add too much text to a card component causing it to break layout.""",
    
    "Sizing Proportion": """Set an image to extreme dimensions (e.g., width: 10px, height: 200px), or make a container unnecessarily huge.
    For example, distort an image aspect ratio, or make a small icon take up entire width.""",
    
    "Responsiveness": """Remove media queries, or set an element to fixed width for all screen sizes.
    For example, remove responsive breakpoints causing layout to break on mobile.""",
    
    "Loss of Interactivity": """Disable a button element, or use CSS pointer-events: none to make a link unclickable.
    For example, add disabled attribute to submit button, or block clicks on navigation links.""",
    
    "Semantic Error": """Replace heading <h1> element with <div> element styled the same way.
    For example, convert semantic nav to div, or replace button with styled span.""",
    
    "Nesting Error": """Place an <a> tag inside another <a> tag, or put a <div> inside a <p> tag.
    For example, nest block elements inside inline elements incorrectly.""",
    
    "Missing Attributes": """Remove alt attribute from <img> elements, or remove aria-label from form inputs.
    For example, remove accessibility attributes, or remove required form attributes.""",
}


class RepairTaskSynthesizer(BaseSynthesizer):
    def generate_defect_task(self, generation_data: Dict, defect_type: str) -> Dict:
        """
        逆向生成策略：
        现有代码(Generation) -> 目标代码（正确的结果）
        LLM注入缺陷 -> 源代码（需要修复的代码）
        用户任务：修复缺陷，从源代码恢复到目标代码
        """
        dst_code_context = self.format_code_context(generation_data["dst_code"])
        defect_description = DEFECT_DESCRIPTIONS.get(defect_type, "")

        prompt =f"""You are an expert web developer. I have a clean, high-quality codebase for a webpage.
I want to generate a dataset for web repair/debugging tasks.

Current Defect Type: {defect_type}

Defect Injection Instructions:
{defect_description}

Please analyze the provided code and inject a specific defect that fits the '{defect_type}' category.
The defect should be realistic and something that could occur during development.
The defect should be visually noticeable or functionally impactful.

Return XML format with the following structure:
<description>A clear, one-sentence instruction for the repair task (e.g., 'Fix the z-index issue causing the modal to be hidden behind content' or 'Restore proper contrast between text and background').</description>
<file path="path/to/file">
The full content of the file with the injected defect
</file>
<file path="path/to/another/file">
The full content of another modified file (if needed)
</file>

Important:
- Only include files that were actually modified to inject the defect.
- The description should be a repair instruction (what needs to be fixed), not what was broken.
- Make sure the defect is obvious enough to be noticed visually or functionally.
- The defect should be fixable by modifying HTML/CSS/JS code.

Here is the clean code (which will be the Goal/Dst state after repair):
{dst_code_context}"""
        try:
            response = self._create_chat_completion_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates web development debugging datasets in XML format.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            result = response
            # dst_code 是正确的代码（目标状态）
            dst_code = generation_data["dst_code"]
            # src_code 是注入缺陷后的代码（需要修复的起始状态）
            src_code = self.merge_code_with_modifications(
                dst_code,
                result.get("modified_files", [])
            )

            return {
                "task": "repair",
                "task_type": [defect_type],
                "description": result["description"],
                "defect_explanation": result.get("defect_explanation", ""),
                "src_code": src_code,  # 有缺陷的代码
                "dst_code": dst_code,  # 正确的代码
                "src_screenshot": [],  # 需要后续生成
                "dst_screenshot": generation_data.get("dst_screenshot", []),
                "modified_files": result.get("modified_files", []),
                "llm_raw_response": result.get("raw_response"),
                "llm_metadata": result.get("llm_metadata"),
            }
        except Exception as e:
            print(f"Error generating defect task ({defect_type}): {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def process_single_generation_entry(
        self,
        generation_entry: Dict,
        output_dir: str = None,
        folder_name: str = None,
        source_generation_dir: str = None,
        defect_types: List[str] = None,
    ) -> List[Dict]:
        """
        处理单个 generation entry，为每种缺陷类型生成修复任务
        
        Args:
            generation_entry: 原始 generation 数据
            output_dir: 输出目录
            folder_name: 文件夹名称
            source_generation_dir: 源 generation 目录
            defect_types: 要生成的缺陷类型列表，默认为所有类型
        """
        generated_tasks = []
        task_index = 0
        
        # 如果没有指定缺陷类型，使用所有类型
        if defect_types is None:
            defect_types = DEFECT_TYPES

        for defect_type in defect_types:
            print(f"Generating Repair Task for Defect: {defect_type}")
            task = self.generate_defect_task(generation_entry, defect_type)
            if task:
                generated_tasks.append(task)
                # 立即保存
                if output_dir and folder_name:
                    defect_type_suffix = defect_type.lower().replace(" ", "_")
                    task_id = f"{folder_name}_{defect_type_suffix}_{task_index}"
                    save_repair_task(
                        task,
                        output_dir,
                        task_id,
                        source_generation_dir=source_generation_dir,
                    )
                task_index += 1

        return generated_tasks


def save_repair_task(
    task: Dict,
    output_base_dir: str,
    task_id: str,
    source_generation_dir: str = None,
) -> None:
    """
    Save a repair task to the filesystem with proper directory structure.

    Structure:
    output_base_dir/
    └── {task_id}/
        ├── src/          # 有缺陷的代码
        │   ├── index.html
        │   └── resources/
        │       └── ...
        ├── dst/          # 正确的代码（修复后）
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

    # Save src code files (有缺陷的代码)
    for file_info in task["src_code"]:
        file_path = os.path.join(src_dir, file_info["directory"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_info["code"])

    # Save dst code files (正确的代码)
    for file_info in task["dst_code"]:
        file_path = os.path.join(dst_dir, file_info["directory"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_info["code"])

    # Prepare info.json
    info = {
        "task": task["task"],
        "task_type": task["task_type"],
        "description": task["description"],
        "defect_explanation": task.get("defect_explanation", ""),
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
    
    print(f"Saved repair task to {task_dir}")


def process_single_info_json(args):
    """
    处理单个 info.json 文件的辅助函数

    Args:
        args: 包含参数的元组

    Returns:
        处理的任务数量
    """
    full_path, original_folder_name, root, output_dir, synthesizer, num_defect_types = args
    
    # 在每个线程中随机选择缺陷类型
    selected_defect_types = random.sample(DEFECT_TYPES, min(num_defect_types, len(DEFECT_TYPES)))

    print(f"Processing {full_path}...")
    print(f"  Selected defect types: {selected_defect_types}")
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            gen_data = json.load(f)

        new_tasks = synthesizer.process_single_generation_entry(
            gen_data,
            output_dir=output_dir,
            folder_name=original_folder_name,
            source_generation_dir=root,
            defect_types=selected_defect_types,
        )
        print(f"✓ Completed {full_path}: {len(new_tasks)} tasks")
        return len(new_tasks)
    except Exception as e:
        print(f"✗ Error processing {full_path}: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main(max_workers=4, num_defect_types=3):
    """
    主函数 - 多线程版本

    Args:
        max_workers: 最大线程数,默认为4
        num_defect_types: 每个任务随机选择的缺陷类型数量，默认为3
    """
    # Configuration
    input_dir = (
        "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/generation"
    )
    output_dir = "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/repair"
    config = OmegaConf.load("config/api.yaml")
    api_key = config.api.api_key
    base_url = config.api.base_url
    model = "claude-3-5-sonnet-coder"

    synthesizer = RepairTaskSynthesizer(api_key, base_url, model, max_tokens=32*1024)

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
                        num_defect_types,  # 传递数量而非已选择的类型
                    )
                )

    print(f"Found {len(tasks_to_process)} generation folders to process")
    print(f"Using {max_workers} worker threads")
    print(f"Each task will randomly select {num_defect_types} defect types")

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
    print(f"Generated {task_counter} repair tasks in {output_dir}")
    print(f"{'='*60}")


def test_single_generation(
    generation_folder: str, 
    output_dir: str = None,
    defect_types: List[str] = None,
) -> List[Dict]:
    """
    测试函数:处理指定的单个 generation 文件夹

    Args:
        generation_folder: generation 文件夹的路径
        output_dir: 输出目录,如果为 None 则不保存文件,只返回结果
        defect_types: 要生成的缺陷类型列表，默认为所有类型

    Returns:
        生成的 repair task 列表

    Example:
        # 只生成不保存
        tasks = test_single_generation("/path/to/generation/folder")

        # 生成并保存，只测试部分缺陷类型
        tasks = test_single_generation(
            "/path/to/generation/folder", 
            "/path/to/output",
            defect_types=["Occlusion", "Color Contrast"]
        )
    """
    config = OmegaConf.load("config/api.yaml")
    api_key = config.api.api_key
    base_url = config.api.base_url
    model = "claude-3-5-sonnet-coder"
    synthesizer = RepairTaskSynthesizer(api_key, base_url, model, max_tokens=16*1024)

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
            defect_types=defect_types,
        )

        print(f"Generated {len(new_tasks)} repair tasks")

        # 打印任务摘要
        for i, task in enumerate(new_tasks):
            print(f"\n--- Task {i+1} ---")
            print(f"Defect Type: {task['task_type']}")
            print(f"Description: {task['description']}")
            print(f"Defect Explanation: {task.get('defect_explanation', 'N/A')}")
            print(
                f"Modified files: {[f['directory'] for f in task.get('modified_files', [])]}"
            )

        return new_tasks

    except Exception as e:
        print(f"Error processing {info_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_single_defect_type(
    generation_folder: str,
    defect_type: str,
    output_dir: str = None,
) -> Dict:
    """
    测试单个缺陷类型的生成

    Args:
        generation_folder: generation 文件夹的路径
        defect_type: 要测试的缺陷类型
        output_dir: 输出目录

    Returns:
        生成的 repair task
    """
    if defect_type not in DEFECT_TYPES:
        print(f"Error: Invalid defect type '{defect_type}'")
        print(f"Valid types: {DEFECT_TYPES}")
        return None
    
    return test_single_generation(
        generation_folder, 
        output_dir, 
        defect_types=[defect_type]
    )


if __name__ == "__main__":
    # 可以调整线程数,建议根据 API 限流情况设置
    # 运行所有缺陷类型
    # main(max_workers=5)
    
    # # 或者只运行部分缺陷类型
    main(max_workers=5, num_defect_types=4)

    # # 或者测试单个文件夹
    # tasks = test_single_generation(
    #     "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/generation/1009769_www.kccworld.co.kr_english_",
    #     "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/repair_test",
    #     defect_types=["Occlusion", "Color Contrast"]
    # )
    
    # 或者测试单个缺陷类型
    # task = test_single_defect_type(
    #     "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/generation/1009769_www.kccworld.co.kr_english_",
    #     "Occlusion",
    #     "/Users/pedestrian/Desktop/web_case/data_demo_renderbench/repair_test"
    # )