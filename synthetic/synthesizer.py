from openai import OpenAI
import os
import json
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


class BaseSynthesizer(ABC):
    """
    基础合成器类，提供通用的LLM交互功能
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 8192,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens

    def format_code_context(self, code_list: List[Dict]) -> str:
        """
        将代码文件列表转换为XML格式

        Args:
            code_list: 包含代码文件信息的字典列表

        Returns:
            XML格式的代码上下文字符串
        """
        context = "<code_context>\n"
        for item in code_list:
            # 转义XML特殊字符
            context += (
                f'<file path="{item["path"]}">\n{item["code"]}\n</file>\n'
            )
        context += "</code_context>"
        return context

    def parse_llm_response(self, response_text: str) -> Dict:
        """
        使用正则表达式解析LLM的XML响应（search/replace格式）

        Args:
            response_text: LLM返回的原始响应文本

        Returns:
            解析后的字典，包含description和modified_files
        """
        # 移除markdown代码块标记
        if "```xml" in response_text:
            response_text = (
                response_text.split("```xml")[1].split("```")[0].strip()
            )
        elif "```" in response_text:
            # 尝试提取第一个代码块
            parts = response_text.split("```")
            if len(parts) >= 3:
                response_text = parts[1]
                # 移除可能的语言标识
                if response_text.startswith(("xml", "XML")):
                    response_text = response_text[3:].strip()

        # 提取描述
        desc_pattern = r"<description>(.*?)</description>"
        desc_match = re.search(desc_pattern, response_text, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ""

        # 提取 search_replace 块
        search_replace_pattern = r'<search_replace\s+path="([^"]+)">\s*<search>(.*?)</search>\s*<replace>(.*?)</replace>\s*</search_replace>'
        sr_matches = re.findall(
            search_replace_pattern, response_text, re.DOTALL
        )

        modified_files = []
        for path, search, replace in sr_matches:
            search_stripped = search.strip()
            replace_stripped = replace.strip()

            # 检查 search 和 replace 是否相同
            if search_stripped == replace_stripped:
                print(
                    f"Invalid modification: search and replace are identical for path {path.strip()}"
                )
                continue

            modified_files.append(
                {
                    "path": path.strip(),
                    "search": search_stripped,
                    "replace": replace_stripped,
                }
            )

        result = {
            "description": description,
            "modified_files": modified_files,
        }

        return result

    def apply_search_replace(
        self, code_list: List[Dict], modified_files: List[Dict]
    ) -> List[Dict]:
        """
        将 search/replace 块应用到源代码

        Args:
            code_list: 原始代码文件列表
            modified_files: search/replace 块列表

        Returns:
            修改后的代码文件列表
        """
        # 创建代码的副本
        result_code = []
        code_map = {item["path"]: item["code"] for item in code_list}

        # 按路径分组 modified_files
        blocks_by_path = {}
        for block in modified_files:
            path = block["path"]
            if path not in blocks_by_path:
                blocks_by_path[path] = []
            blocks_by_path[path].append(block)

        # 应用每个文件的修改
        for path, blocks in blocks_by_path.items():
            if path in code_map:
                code = code_map[path]
                for block in blocks:
                    search_text = block["search"]
                    replace_text = block["replace"]

                    if search_text in code:
                        code = code.replace(search_text, replace_text, 1)
                    else:
                        # 尝试忽略空白差异的匹配
                        normalized_search = self._normalize_whitespace(
                            search_text
                        )
                        lines = code.split("\n")
                        found = False

                        for i in range(len(lines)):
                            # 尝试匹配连续的行
                            for j in range(i + 1, len(lines) + 1):
                                candidate = "\n".join(lines[i:j])
                                if (
                                    self._normalize_whitespace(candidate)
                                    == normalized_search
                                ):
                                    code = code.replace(
                                        candidate, replace_text, 1
                                    )
                                    found = True
                                    break
                            if found:
                                break

                        if not found:
                            print(
                                f"Warning: Could not find search text in {path}"
                            )
                            print(f"Search text: {search_text[:100]}...")

                code_map[path] = code

        # 构建结果列表
        for item in code_list:
            new_item = item.copy()
            if item["path"] in code_map:
                new_item["code"] = code_map[item["path"]]
            result_code.append(new_item)

        return result_code

    def _normalize_whitespace(self, text: str) -> str:
        """
        标准化空白字符以进行模糊匹配
        """
        # 移除行尾空白，标准化换行
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    def _create_chat_completion_with_retry(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 3,
        backoff_base: int = 2,
    ) -> Dict:
        """
        带重试机制的聊天完成调用，包含立即解析
        记录所有原始响应，包括失败的尝试，用于调试

        Args:
            messages: 消息列表
            max_retries: 最大重试次数
            backoff_base: 退避基数

        Returns:
            解析后的响应字典，包含raw_response和llm_metadata
        """
        last_error = None
        last_raw_response = None
        all_attempts = []  # 记录所有尝试用于调试

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                )

                response_text = (
                    response.choices[0].message.content
                    if response and response.choices
                    else None
                )
                if not response_text:
                    raise ValueError("Empty response content from LLM.")
                last_raw_response = response_text

                # 记录这次尝试
                attempt_record = {
                    "attempt": attempt,
                    "raw_response": response_text,
                    "success": False,
                    "error": None,
                }

                parsed = self.parse_llm_response(response_text)
                if not parsed:
                    attempt_record["error"] = "Failed to parse LLM response"
                    all_attempts.append(attempt_record)
                    raise ValueError("Failed to parse LLM response.")
                if not parsed.get("modified_files"):
                    attempt_record["error"] = (
                        "Parsed LLM response has no modified_files"
                    )
                    all_attempts.append(attempt_record)
                    raise ValueError(
                        "Parsed LLM response has no modified_files."
                    )

                # 成功！
                attempt_record["success"] = True
                all_attempts.append(attempt_record)

                parsed["raw_response"] = response_text
                parsed["llm_metadata"] = {
                    "model": self.model,
                    "total_attempts": attempt,
                    "all_attempts": all_attempts,  # 包含所有尝试用于调试
                }
                return parsed
            except Exception as exc:
                last_error = exc

                # 如果还没记录这次尝试（如API错误），现在记录
                if not all_attempts or all_attempts[-1]["attempt"] != attempt:
                    all_attempts.append(
                        {
                            "attempt": attempt,
                            "raw_response": last_raw_response,
                            "success": False,
                            "error": str(exc),
                        }
                    )

                wait_time = backoff_base**attempt
                snippet = (
                    (last_raw_response[:200] + "...")
                    if last_raw_response and len(last_raw_response) > 200
                    else last_raw_response
                )
                if attempt == max_retries:
                    print(
                        f"Chat completion failed after {attempt} attempts. "
                        f"Last raw response snippet: {snippet}"
                    )
                    raise
                print(
                    f"Chat completion failed (attempt {attempt}/{max_retries}): {exc}. "
                    f"Retrying in {wait_time}s... Last raw response snippet: {snippet}"
                )
                time.sleep(wait_time)
        raise last_error

    @abstractmethod
    def process_single_generation_entry(
        self,
        generation_entry: Dict,
        output_dir: str = None,
        folder_name: str = None,
        source_generation_dir: str = None,
        task_types: List[str] = None,
        difficulty_levels: List[int] = None,
    ) -> List[Dict]:
        """
        处理单个生成条目的抽象方法

        Args:
            generation_entry: 生成数据条目
            output_dir: 输出目录
            folder_name: 文件夹名称
            source_generation_dir: 源生成目录
            task_types: 要生成的任务类型列表
            difficulty_levels: 难度等级列表（即每个任务包含的任务类型数量）

        Returns:
            生成的任务列表
        """
        pass

    def _process_single_info_json(
        self,
        args: Tuple,
    ) -> int:
        """
        处理单个 info.json 文件的辅助函数

        Args:
            args: 包含参数的元组 (full_path, original_folder_name, root, output_dir, task_types, difficulty_levels)

        Returns:
            处理的任务数量
        """
        (
            full_path,
            original_folder_name,
            root,
            output_dir,
            task_types,
            difficulty_levels,
        ) = args

        print(f"Processing {full_path}...")
        if task_types:
            print(f"  Task types: {task_types}")
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                gen_data = json.load(f)

            new_tasks = self.process_single_generation_entry(
                gen_data,
                output_dir=output_dir,
                folder_name=original_folder_name,
                source_generation_dir=root,
                task_types=task_types,
                difficulty_levels=difficulty_levels,
            )
            print(f"✓ Completed {full_path}: {len(new_tasks)} tasks")
            return len(new_tasks)
        except Exception as e:
            print(f"✗ Error processing {full_path}: {e}")
            import traceback

            traceback.print_exc()
            return 0

    def run_batch_processing(
        self,
        input_dir: str,
        output_dir: str,
        max_workers: int = 4,
        task_types: List[str] = None,
        difficulty_levels: List[int] = None,
    ) -> int:
        """
        批量处理 generation 文件夹的多线程函数

        Args:
            input_dir: 输入目录，包含 generation 文件夹
            output_dir: 输出目录
            max_workers: 最大线程数，默认为4
            task_types: 可选的任务类型列表，用于过滤任务
            difficulty_levels: 难度等级列表（即每个任务包含的任务类型数量），例如 [1, 2, 3]
        Returns:
            生成的任务总数
        """
        import random

        tasks_to_process = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file == "info.json":
                    full_path = os.path.join(root, file)
                    original_folder_name = os.path.basename(root)

                    # 这里不再做随机抽取，而是将配置传递给处理函数
                    # 具体的随机组合逻辑在 process_single_generation_entry 中实现

                    tasks_to_process.append(
                        (
                            full_path,
                            original_folder_name,
                            root,
                            output_dir,
                            task_types,
                            difficulty_levels,
                        )
                    )

        print(f"Found {len(tasks_to_process)} generation folders to process")
        print(f"Using {max_workers} worker threads")

        task_counter = 0

        # 使用线程池处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_path = {
                executor.submit(
                    self._process_single_info_json, task_args
                ): task_args[0]
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
        print(f"Generated {task_counter} tasks in {output_dir}")
        print(f"{'='*60}")

        return task_counter

    def test_single_generation(
        self,
        generation_folder: str,
        output_dir: str = None,
        task_types: List[str] = None,
    ) -> List[Dict]:
        """
        测试函数：处理指定的单个 generation 文件夹

        Args:
            generation_folder: generation 文件夹的路径
            output_dir: 输出目录，如果为 None 则不保存文件，只返回结果
            task_types: 要生成的任务类型列表

        Returns:
            生成的 task 列表
        """
        info_path = os.path.join(generation_folder, "info.json")

        if not os.path.exists(info_path):
            print(f"Error: info.json not found in {generation_folder}")
            return []

        original_folder_name = os.path.basename(generation_folder)
        print(f"Processing {info_path}...")

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                gen_data = json.load(f)

            new_tasks = self.process_single_generation_entry(
                gen_data,
                output_dir=output_dir,
                folder_name=original_folder_name,
                source_generation_dir=generation_folder,
                task_types=task_types,
            )

            print(f"Generated {len(new_tasks)} tasks")

            # 打印任务摘要
            for i, task in enumerate(new_tasks):
                print(f"\n--- Task {i+1} ---")
                print(f"Type: {task['task_type']}")
                print(f"Description: {task['description']}")
                print(
                    f"Modified files: {[f['path'] for f in task.get('label_modified_files', [])]}"
                )

            return new_tasks

        except Exception as e:
            print(f"Error processing {info_path}: {e}")
            import traceback

            traceback.print_exc()
            return []
