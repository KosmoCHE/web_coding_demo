from openai import OpenAI
import os
import json
from typing import List, Dict, Any
from omegaconf import OmegaConf
import shutil
import time
import random
from synthetic.synthesizer import BaseSynthesizer
from utils.config import *
from utils.utils import save_task

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
    
    def generate_forward_task(self, generation_data: Dict, task_types: List[str]) -> Dict:
        """
        策略1：前向演化 - 支持多任务类型组合
        现有代码(Generation) -> 源代码
        LLM修改它 -> 目标代码
        """
        src_code = generation_data["dst_code"]
        src_code_context = self.format_code_context(src_code)
        
        # 构建多任务描述
        task_descriptions_str = ""
        for t_type in task_types:
            desc = TASK_DESCRIPTIONS.get(t_type, "")
            task_descriptions_str += f"- Task Type: {t_type}\n  Guideline: {desc}\n\n"

        prompt = f"""You are an expert web developer. I have a codebase for a webpage.
I want to generate a dataset for web editing tasks.

Current Task Types (Combined Task): {', '.join(task_types)}

Task Instructions:
You must perform modifications corresponding to ALL the following task types in a single pass:

{task_descriptions_str}

Please analyze the provided code and propose a specific, reasonable modification task that combines ALL the requested changes.
Then, implement the modifications using search/replace blocks.

Return XML format with the following structure:
<description>
A instruction for ALL the requested task types.
It must clearly identify the target element (e.g., by its text content, position, or unique feature) so the intent is unambiguous.
However, it must NOT reveal the exact code implementation details (e.g., do not mention specific class names, ID selectors, or exact CSS property values unless they are part of the requirement).
Example: 'Change the background color of the "Sign Up" button to blue' is good. 'Change .btn-primary background to #0000FF' is bad.
</description>
<search_replace path="path/to/file">
<search>
exact text to find in the original file
</search>
<replace>
replacement text with the modification applied
</replace>
</search_replace>

Important:
- You MUST implement modifications for ALL requested task types.
- The <search> block must contain the EXACT text from the original file (including whitespace and indentation).
- The <replace> block contains the modified code.
- One <search_replace> block can only contain one pair of <search> and <replace>.
- You can include multiple <search_replace> blocks if you need to modify multiple locations, you can also modify multiple files.
- Make sure the modifications are obvious enough in visual appearance.

Here is the source code:
{src_code_context}"""

        try:
            result = self._generate_and_apply_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates web development datasets in XML format using search/replace blocks.",
                    },
                    {"role": "user", "content": prompt},
                ],
                code_list=src_code,
                max_retries=self.max_retries,
            )

            # dst_code 是应用修改后的代码
            dst_code = result["modified_code"]

            return {
                "task": "edit",
                "task_type": task_types,
                "description": result["description"],
                "src_code": src_code,
                "dst_code": dst_code,
                "label_modified_files": result.get("modified_files", []),  # 训练时src2dst时应该使用的modified_files
                "resources": generation_data.get("resources", []),
                "synthetic_modified_files": result.get("modified_files", []),  # 合成数据时进行修改的文件
                "llm_raw_response": result.get("raw_response"),
                "llm_metadata": result.get("llm_metadata"),
            }
        except Exception as e:
            print(f"Error generating forward task ({task_types}): {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_single_generation_entry(
        self,
        generation_entry: Dict,
        output_dir: str = None,
        folder_name: str = None,
        source_generation_dir: str = None,
        task_types: List[str] = FORWARD_TASKS,
        difficulty_levels: List[int] = [1],
    ) -> List[Dict]:
        """
        处理单个 generation entry，为每种任务类型生成编辑任务
        
        Args:
            generation_entry: 原始 generation 数据
            output_dir: 输出目录
            folder_name: 文件夹名称
            source_generation_dir: 源 generation 目录
            task_types: 要生成的前向任务类型列表，默认为所有类型
        """
        generated_tasks = []
        task_index = 0


        # 遍历每个难度等级（即组合的任务数量）
        for level in difficulty_levels:
            # 确保不超过可用的任务类型总数
            if level > len(task_types):
                continue
                
            # 随机抽取 level 个任务类型
            selected_types = random.sample(task_types, level)
            
            print(f"Generating Forward Task (Level {level}): {selected_types}")
            task = self.generate_forward_task(generation_entry, selected_types)
            
            if task:
                generated_tasks.append(task)
                # 立即保存
                if output_dir and folder_name:
                    # 文件名包含难度等级
                    task_id = f"{folder_name}_L{level}_{task_index}"
                    save_task(
                        task,
                        output_dir,
                        task_id,
                        source_generation_dir=source_generation_dir,
                    )
                task_index += 1

        return generated_tasks


def main(max_workers=4, difficulty_levels=None, max_retries=3):
    """
    主函数 - 多线程版本

    Args:
        max_workers: 最大线程数
        difficulty_levels: 难度等级列表，例如 [1, 2, 3] 表示分别生成包含1个、2个、3个修改的任务
    """
    # Configuration
    input_dir = (
        "/Users/pedestrian/Desktop/web_case/data/data_demo_renderbench/generation"
    )
    output_dir = "/Users/pedestrian/Desktop/web_case/data/data_demo_renderbench/edit"
    config = OmegaConf.load("config/api.yaml")
    api_key = config.api.api_key
    base_url = config.api.base_url
    model = "gpt-5-codex"

    synthesizer = EditTaskSynthesizer(api_key, base_url, model, max_tokens=32*1024, max_retries=max_retries)

    if difficulty_levels is None:
        difficulty_levels = [1, 2, 3] # 默认生成 1, 2, 3 种修改组合的任务
        
    print(f"Generating tasks with difficulty levels: {difficulty_levels}")

    synthesizer.run_batch_processing(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=max_workers,
        task_types=FORWARD_TASKS,
        difficulty_levels=difficulty_levels,
    )


def test_single_generation(
    generation_folder: str, 
    output_dir: str = None,
    task_types: List[str] = None,
    difficulty_levels: List[int] = None,
) -> List[Dict]:
    """
    测试函数:处理指定的单个 generation 文件夹

    Args:
        generation_folder: generation 文件夹的路径
        output_dir: 输出目录,如果为 None 则不保存文件,只返回结果
        task_types: 要生成的前向任务类型列表，默认为所有类型
        difficulty_levels: 难度等级列表，例如 [1, 2, 3] 表示分别生成包含1个、2个、3个修改的任务

    Returns:
        生成的 edit task 列表

    Example:
        # 只生成不保存
        tasks = test_single_generation("/path/to/generation/folder")

        # 生成并保存，只测试部分任务类型
        tasks = test_single_generation(
            "/path/to/generation/folder", 
            "/path/to/output",
            task_types=["Style Modification"]
        )
    """
    config = OmegaConf.load("config/api.yaml")
    api_key = config.api.api_key
    base_url = config.api.base_url
    model = "gpt-5-codex"
    synthesizer = EditTaskSynthesizer(api_key, base_url, model, max_tokens=16*1024)
    
    # 模拟 process_single_generation_entry 的调用逻辑
    info_path = os.path.join(generation_folder, "info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        gen_data = json.load(f)
        
    return synthesizer.process_single_generation_entry(
        gen_data,
        output_dir=output_dir,
        folder_name=os.path.basename(generation_folder),
        source_generation_dir=generation_folder,
        task_types=task_types,
        difficulty_levels=difficulty_levels or [1, 2]
    )

if __name__ == "__main__":
    # main(max_workers=5, difficulty_levels=[1, 3], max_retries=6)

    # 或者测试单个文件夹
    tasks = test_single_generation(
        "/Users/pedestrian/Desktop/web_case/data/data_demo_renderbench/generation/1009769_www.kccworld.co.kr_english_",
        "/Users/pedestrian/Desktop/web_case/data/data_demo_renderbench/edit_test_multi",
        task_types=["Style Modification", "Add Element", "Function Modification"],
        difficulty_levels=[1], # 测试生成包含2个修改的任务
    )

