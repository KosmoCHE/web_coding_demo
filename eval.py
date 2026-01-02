import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal
from omegaconf import OmegaConf
from tqdm import tqdm

from mllm.openai_chat import OpenAIChat
from mllm.prompt import (
    Generation_Instruction_Prompt,
    Edit_Instruction_Prompt,
    Repair_Instruction_Prompt,
)


def get_task_folders(base_path: str, task_type: str) -> List[str]:
    """获取指定任务类型的所有文件夹"""
    task_path = Path(base_path) / task_type
    if not task_path.exists():
        print(f"Warning: {task_type} path not found: {task_path}")
        return []

    folders = [str(f) for f in task_path.iterdir() if f.is_dir()]
    return folders


def process_single_task(
    data_folder: str,
    client: OpenAIChat,
    task_type: str,
    mode: str = "image",
) -> dict:
    """处理单个任务"""
    try:
        if task_type == "generation":
            result = client.run_generation_task(
                data_folder=data_folder,
                mode=mode,
                instruction_prompt=Generation_Instruction_Prompt,
            )
        elif task_type == "edit":
            result = client.run_edit_repair_task(
                data_folder=data_folder,
                mode=mode,
                instruction_prompt=Edit_Instruction_Prompt,
            )
        elif task_type == "repair":
            result = client.run_edit_repair_task(
                data_folder=data_folder,
                mode=mode,
                instruction_prompt=Repair_Instruction_Prompt,
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return {
            "folder": data_folder,
            "task_type": task_type,
            "status": "success",
            "result": result,
        }
    except Exception as e:
        return {
            "folder": data_folder,
            "task_type": task_type,
            "status": "failed",
            "error": str(e),
        }


def eval_single_task_type(
    base_path: str,
    task_type: str,
    client: OpenAIChat,
    mode: str = "image",
    max_workers: int = 8,
) -> dict:
    """并行评估单个任务类型的所有任务

    Args:
        base_path: 数据集基础路径
        task_type: 任务类型 (generation/edit/repair)
        client: OpenAI 客户端
        mode: 模式 ("image" 或 "text")
        max_workers: 最大线程数

    Returns:
        包含结果和统计的字典
    """
    print(f"\n{'='*60}")
    print(f"Processing {task_type.upper()} tasks")
    print(f"{'='*60}")

    # 获取该任务类型的所有文件夹
    folders = get_task_folders(base_path, task_type)
    total = len(folders)

    if total == 0:
        print(f"No {task_type} tasks found")
        return {
            "task_type": task_type,
            "total": 0,
            "success": 0,
            "failed": 0,
            "results": [],
        }

    print(f"Found {total} {task_type} tasks\n")

    results = []
    success_count = 0
    failed_count = 0

    # 使用线程池并行处理该任务类型
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_folder = {
            executor.submit(
                process_single_task, folder, client, task_type, mode
            ): folder
            for folder in folders
        }

        # 使用tqdm显示进度
        with tqdm(total=total, desc=f"{task_type.capitalize()}") as pbar:
            for future in as_completed(future_to_folder):
                result = future.result()
                results.append(result)

                if result["status"] == "success":
                    success_count += 1
                else:
                    failed_count += 1
                    print(f"\nFailed: {result['folder']}")
                    print(f"Error: {result['error']}")

                pbar.update(1)
                pbar.set_postfix(success=success_count, failed=failed_count)

    # 输出该任务类型的统计
    success_rate = (success_count / total * 100) if total > 0 else 0
    print(f"\n{task_type.upper()} Summary:")
    print(f"  Total: {total}")
    print(f"  Success: {success_count} ({success_rate:.2f}%)")
    print(f"  Failed: {failed_count}")

    return {
        "task_type": task_type,
        "total": total,
        "success": success_count,
        "failed": failed_count,
        "results": results,
    }


if __name__ == "__main__":
    # 配置参数
    BASE_PATH = "/Users/pedestrian/Desktop/web_case/data/data_demo_renderbench_10"
    CONFIG_PATH = "config/api.yaml"
    MODE = "image"  # 或 "text"
    MAX_WORKERS = 5  # 根据你的API限制调整
    # TASK_TYPES = ["generation", "edit", "repair"]  # 要测试的任务类型
    TASK_TYPES = ["repair"]  # 要测试的任务类型

    # 加载配置
    config = OmegaConf.load(CONFIG_PATH)
    api_key = config.api.api_key
    base_url = config.api.base_url
    model = "gpt-5-codex"

    # 创建客户端
    client = OpenAIChat(
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=32 * 1024,
        max_retry=6,
    )

    # 按任务类型顺序执行，每个类型内部并行
    all_results = []
    overall_stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
    }

    for task_type in TASK_TYPES:
        task_result = eval_single_task_type(
            base_path=BASE_PATH,
            task_type=task_type,
            client=client,
            mode=MODE,
            max_workers=MAX_WORKERS,
        )
        all_results.append(task_result)

        overall_stats["total"] += task_result["total"]
        overall_stats["success"] += task_result["success"]
        overall_stats["failed"] += task_result["failed"]

    # 输出总体统计
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {overall_stats['total']}")
    print(f"Total success: {overall_stats['success']}")
    print(f"Total failed: {overall_stats['failed']}")

    if overall_stats["total"] > 0:
        overall_rate = overall_stats["success"] / overall_stats["total"] * 100
        print(f"Overall success rate: {overall_rate:.2f}%")

    print("\nBreakdown by task type:")
    for task_result in all_results:
        if task_result["total"] > 0:
            rate = task_result["success"] / task_result["total"] * 100
            print(
                f"  {task_result['task_type']}: {task_result['success']}/{task_result['total']} ({rate:.2f}%)"
            )

    # 保存失败的任务列表
    all_failed = []
    for task_result in all_results:
        failed = [r for r in task_result["results"] if r["status"] == "failed"]
        all_failed.extend(failed)

    if all_failed:
        output_file = "failed_tasks.txt"
        with open(output_file, "w") as f:
            for r in all_failed:
                f.write(f"[{r['task_type']}] {r['folder']}\n")
                f.write(f"  Error: {r['error']}\n\n")
        print(f"\nFailed tasks saved to: {output_file}")
