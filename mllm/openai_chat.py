import os
from openai import OpenAI
from typing import List, Optional

from mllm.base import MLLMChat
from utils.utils import chat_with_retry
class OpenAIChat(MLLMChat):
    def __init__(self, model_name: str, client: Optional[OpenAI] = None, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs) -> None:
        # 如果提供了client就使用client,否则创建新的client
        if client:
            self.client = client
        else:
            # 优先使用传入的参数,其次使用环境变量
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)
        
        super().__init__(model_name, **kwargs)
    
    def chat(self, messages: List[dict], max_retries: int = 3) -> str:
        """
        发送消息到OpenAI并获取响应
        Args:
            messages: OpenAI格式的消息列表
            max_retries: 最大重试次数
        Returns:
            LLM的响应内容字符串
        """
        response = chat_with_retry(
            client=self.client,
            messages=messages,
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            max_retries=max_retries
        )
        
        return response
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    from mllm.prompt import *
    config = OmegaConf.load("config/api.yaml")
    api_key = config.api.api_key
    base_url = config.api.base_url
    model = "gpt-5-codex"
    client = OpenAIChat(
        model_name=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=32 * 1024,
    )
    # 测试单个generation文件夹
    # data_folder = "/Users/pedestrian/Desktop/web_case/data/data_demo_renderbench/generation/2931255_www.testing.com"
    # client.run_generation_task(
    #     data_folder=data_folder,
    #     mode="image",
    #     instruction_prompt=Generation_Instruction_Prompt,
    # )
    
    
    # 测试edit 任务
    data_folder = "/Users/pedestrian/Desktop/web_case/data/data_demo_renderbench_v1/edit_test_multi/1009769_www.kccworld.co.kr_english__L2_0"
    client.run_edit_repair_task(
        data_folder=data_folder,
        mode="text",
        instruction_prompt=Edit_Instruction_Prompt,
    )