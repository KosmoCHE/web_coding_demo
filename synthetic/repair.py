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

# 破坏类型分类
DEFECT_TYPES = [
    "Occlusion",           # 遮挡
    "Crowding",            # 拥挤
    "Text Overlap",        # 文本重叠
    "Alignment",           # 对齐问题
    "Color Contrast",      # 颜色与对比度
    "Overflow",            # 溢出
    "Sizing Proportion",   # 尺寸/比例失衡
    # "Responsiveness",      # 响应式缺陷
    # "Loss of Interactivity",  # 交互性丧失
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
    def generate_defect_task(self, generation_data: Dict, defect_types: List[str]) -> Dict:
        """
        逆向生成策略：支持多缺陷注入
        现有代码(Generation) -> 目标代码（正确的结果）
        LLM注入缺陷 -> 源代码（需要修复的代码）
        """
        dst_code = generation_data["dst_code"]
        dst_code_context = self.format_code_context(dst_code)
        
        # 构建多缺陷描述
        defect_descriptions_str = ""
        for d_type in defect_types:
            desc = DEFECT_DESCRIPTIONS.get(d_type, "")
            defect_descriptions_str += f"- Defect Type: {d_type}\n  Guideline: {desc}\n\n"

        prompt = f"""You are an expert web developer. I have a clean, high-quality codebase for a webpage.
I want to generate a dataset for web repair/debugging tasks.

Current Defect Types (Combined Defects): {', '.join(defect_types)}

Defect Injection Instructions:
You must inject defects corresponding to ALL the following types in a single pass:

{defect_descriptions_str}

Please analyze the provided code and inject specific defects that fit the requested categories.
The defects should be realistic and something that could occur during development.

Return XML format with the following structure:
<description>
A instruction for ALL the requested task types.
It must clearly identify the issue or the target element (e.g., by its text content, position, or unique feature) so the intent is unambiguous.
However, it must NOT reveal the exact code implementation details (e.g., do not mention specific class names, ID selectors, or exact CSS property values).
Example: 'Fix the "Submit" button being covered by the footer' is good. 'Change z-index of .footer to -1' is bad.
</description>
<search_replace path="path/to/file">
<search>
exact text to find in the original file
</search>
<replace>
replacement text with the defect injected
</replace>
</search_replace>

Important:
- You MUST implement modifications for ALL requested task types.
- The <search> block must contain the EXACT text from the original file.
- The <replace> block contains the modified code with the defect injected.
- One <search_replace> block can only contain one pair of <search> and <replace>.
- You can include multiple <search_replace> blocks if you need to modify multiple locations, you can also modify multiple files.
- The description should be a repair instruction (what needs to be fixed).
- Make sure the defects are obvious enough to be noticed visually or functionally.

Here is the clean code (which will be the Goal/Dst state after repair):
{dst_code_context}"""
        try:
            result = self._generate_and_apply_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates web development debugging datasets in XML format using search/replace blocks.",
                    },
                    {"role": "user", "content": prompt},
                ],
                code_list=dst_code,
                max_retries=self.max_retries,
            )

            # src_code 是注入缺陷后的代码（需要修复的起始状态）
            src_code = result["modified_code"]
            
            label_modified_files = []
            for mod in result.get("modified_files", []):
                label_modified_files.append({
                    "path": mod["path"],
                    "search": mod["replace"],
                    "replace": mod["search"],
                })
            return {
                "task": "repair",
                "task_type": defect_types,
                "description": result["description"],
                "src_code": src_code,  # 有缺陷的代码
                "dst_code": dst_code,  # 正确的代码
                "resources": generation_data.get("resources", []),
                "label_modified_files": label_modified_files,  # 训练时src2dst时应该使用的modified_files
                "synthetic_modified_files": result.get("modified_files", []),  # 合成数据时进行修改的文件
                "llm_raw_response": result.get("raw_response"),
                "llm_metadata": result.get("llm_metadata"),
            }
        except Exception as e:
            print(f"Error generating defect task ({defect_types}): {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def process_single_generation_entry(
        self,
        generation_entry: Dict,
        output_dir: str = None,
        folder_name: str = None,
        source_generation_dir: str = None,
        task_types: List[str] = DEFECT_TYPES,
        difficulty_levels: List[int] = [1],
    ) -> List[Dict]:
        """
        处理单个 generation entry，为每种缺陷类型生成修复任务
        
        Args:
            generation_entry: 原始 generation 数据
            output_dir: 输出目录
            folder_name: 文件夹名称
            source_generation_dir: 源 generation 目录
            task_types: 要生成的缺陷类型列表，默认为所有类型
        """
        generated_tasks = []
        task_index = 0
        
        # 遍历每个难度等级
        for level in difficulty_levels:
            if level > len(task_types):
                continue
                
            # 随机抽取 level 个缺陷类型
            selected_types = random.sample(task_types, level)
            
            print(f"Generating Repair Task (Level {level}): {selected_types}")
            task = self.generate_defect_task(generation_entry, selected_types)
            
            if task:
                generated_tasks.append(task)
                # 立即保存
                if output_dir and folder_name:
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
    """
    # Configuration
    input_dir = (
        "/Users/pedestrian/Desktop/web_case//data/data_demo_renderbench/generation"
    )
    output_dir = "/Users/pedestrian/Desktop/web_case/data/data_demo_renderbench_10/repair"
    config = OmegaConf.load("config/api.yaml")
    api_key = config.api.api_key
    base_url = config.api.base_url
    model = "gpt-5-codex"

    synthesizer = RepairTaskSynthesizer(api_key, base_url, model, max_tokens=32*1024, max_retries=max_retries)

    if difficulty_levels is None:
        difficulty_levels = [1, 2, 3]

    print(f"Generating tasks with difficulty levels: {difficulty_levels}")

    synthesizer.run_batch_processing(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=max_workers,
        task_types=DEFECT_TYPES,
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
        task_types: 要生成的缺陷类型列表，默认为所有类型
        difficulty_levels: 难度等级列表, 默认为 [1, 2]

    Returns:
        生成的 repair task 列表

    Example:
        # 只生成不保存
        tasks = test_single_generation("/path/to/generation/folder")

        # 生成并保存，只测试部分缺陷类型
        tasks = test_single_generation(
            "/path/to/generation/folder", 
            "/path/to/output",
            task_types=["Occlusion", "Color Contrast"]
        )
    """
    config = OmegaConf.load("config/api.yaml")
    api_key = config.api.api_key
    base_url = config.api.base_url
    model = "claude-3-5-sonnet-coder"
    synthesizer = RepairTaskSynthesizer(api_key, base_url, model, max_tokens=16*1024)

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
    main(max_workers=5, difficulty_levels=[3], max_retries=6)
    
    # # 或者测试单个缺陷类型
    # task = test_single_generation(
    #     "/Users/pedestrian/Desktop/web_case/data/data_demo_renderbench/generation/1009769_www.kccworld.co.kr_english_",
    #     "/Users/pedestrian/Desktop/web_case/data/data_demo_renderbench/repair_test_multi",
    #     task_types=["Occlusion", "Color Contrast", "Text Overlap"],
    #     difficulty_levels=[2, 3],
    # )