import base64
from typing import Union, List, Dict, Any, Optional, Tuple
import os
import shutil
import json
import re
from pathlib import Path
from utils.config import CODE_EXTENSIONS, IMAGE_EXTENSIONS
from utils.utils import encode_image, save_screenshots, apply_search_replace, get_image_mime_type

class MLLMChat:
    DEFAULT_MAX_TOKENS = 8192 * 2
    DEFAULT_TEMPERATURE = 0
    DEFAULT_SEED = 42
    
    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.max_tokens = kwargs.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        self.temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
        self.seed = kwargs.get("seed", self.DEFAULT_SEED)
        print(f"Temperature: {self.temperature}, Max Tokens: {self.max_tokens}, Seed: {self.seed}")
        

    def chat(self, messages: List[Dict[str, Any]], max_retries: int = 3) -> str:
        """
        发送消息到LLM并获取响应
        
        Args:
            messages: OpenAI格式的消息列表
            max_retries: 最大重试次数
        
        Returns:
            LLM的响应内容字符串
        """
        
        pass
    
    def load_generation_data(self, data_folder: Path) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        加载 generation 任务的 info.json 数据
        
        Args:
            data_folder: generation 任务的数据文件夹路径
        
        Returns:
            Tuple[str, List[str], List[Dict]]: (description, screenshot_paths, resources_info)
            - description: 任务描述
            - screenshot_paths: 目标截图的完整路径列表
            - resources_info: 资源文件信息列表,包含路径、类型和描述
        """
        info_path = data_folder / "info.json"
        
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        # 1. 获取 description
        description = info.get("description", "")
        
        # 2. 获取 dst_screenshot 路径
        dst_screenshots = info.get("dst_screenshot", [])
        screenshot_paths = [str(data_folder / "dst" / screenshot) for screenshot in dst_screenshots]
        
        # 3. 处理 resources 信息
        resources = info.get("resources", [])
        resources_info = []
        
        for resource in resources:
            resource_type = resource.get("type", "")
            resource_path = resource.get("path", "")
            
            resource_info = {
                "type": resource_type,
                "path": resource_path
            }
            
            # 如果是图片,添加描述信息
            if resource_type == "image":
                resource_info["description"] = resource.get("description", "")
            
            resources_info.append(resource_info)
        
        return description, screenshot_paths, resources_info
    
    def load_edit_repair_data(self, data_folder: Path) -> Tuple[str, List[Dict[str, str]], List[str], List[str], List[Dict[str, Any]]]:
        """
        加载 edit/repair 任务的 info.json 数据
        
        Args:
            data_folder: edit/repair 任务的数据文件夹路径
        
        Returns:
            Tuple[str, List[Dict[str, str]], List[str], List[str]]: (description, src_code, src_screenshots, dst_screenshots)
            - description: 任务描述
            - src_code: 源代码文件列表,格式为 [{"path": "...", "code": "..."}]
            - src_screenshots: 源截图的完整路径列表
            - dst_screenshots: 目标截图的完整路径列表
            - resources_info: 资源文件信息列表,包含路径、类型和描述
        """
        info_path = data_folder / "info.json"
        
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        # 1. 获取 description
        description = info.get("description", "")
        
        # 2. 获取 src 代码
        src_code_list = info.get("src_code", [])
        src_code = []
        
        for file_info in src_code_list:
            file_path = file_info.get("path", "")
            file_code = file_info.get("code", "")
            src_code.append({"path": file_path, "code": file_code})
        
        # 3. 获取 src_screenshot 路径
        src_screenshots_data = info.get("src_screenshot", [])
        src_screenshots = [str(data_folder / "src" / screenshot) for screenshot in src_screenshots_data]
        
        # 4. 获取 dst_screenshot 路径
        dst_screenshots_data = info.get("dst_screenshot", [])
        dst_screenshots = [str(data_folder / "dst" / screenshot) for screenshot in dst_screenshots_data]
        
        # 5. 处理 resources 信息
        resources = info.get("resources", [])
        resources_info = []
        
        for resource in resources:
            resource_type = resource.get("type", "")
            resource_path = resource.get("path", "")
            
            resource_info = {
                "type": resource_type,
                "path": resource_path
            }
            
            # 如果是图片,添加描述信息
            if resource_type == "image":
                resource_info["description"] = resource.get("description", "")
            
            resources_info.append(resource_info)
        
        return description, src_code, src_screenshots, dst_screenshots, resources_info
    
    def create_workspace(self, data_folder: Path, workspace_path: Path, resources_info: List[Dict[str, Any]]) -> None:
        """
        创建工作空间,根据 resources_info 从 dst 中复制资源文件
        
        Args:
            data_folder: generation任务的数据文件夹路径
            workspace_path: 工作空间路径
            resources_info: 资源文件信息列表
        """
        dst_dir = data_folder / "dst"
        
        # 创建workspace目录
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        if not dst_dir.exists():
            return
        
        # 根据 resources_info 复制资源文件
        for resource in resources_info:
            resource_path = resource.get("path", "")
            
            if resource_path:
                # 源文件路径
                src_file = dst_dir / resource_path
                
                if src_file.exists():
                    # 目标路径
                    dst_file = workspace_path / resource_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 复制文件
                    shutil.copy2(src_file, dst_file)
    
    def construct_messages_for_generation(self, 
                                          description: str, 
                                          screenshot_paths: List[str],
                                          resources_info: List[Dict[str, Any]], 
                                          instruction_prompt: str,
                                          mode: str = "text",
                                          workspace_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        
        """
        根据不同模态组装 generation 任务的 messages
        
        Args:
            description: 任务描述
            screenshot_paths: 目标截图的完整路径列表
            resources_info: 资源文件信息列表
            instruction_prompt: 任务提示词
            mode: 模态类型,"text" 或 "image"
            workspace_path: 工作空间路径(image模式需要)
        
        Returns:
            List[Dict[str, Any]]: OpenAI 格式的 messages 列表
        """
        messages = []
        
        # 构建 user message 的 content
        user_content = []
        
        # 1. 首先添加任务指令
        task_instruction = instruction_prompt +"\n"+ f"## Website Description\n{description}\n"
        
        # 2. 添加目标截图说明(仅在 image 模式下)
        if mode == "image" and screenshot_paths:
            task_instruction += "\n## Target Screenshots\n"
            task_instruction += "The following screenshots show the expected result:\n\n"
        
        # 添加任务指令文本
        user_content.append({
            "type": "text",
            "text": task_instruction
        })
        
        # 3. 添加目标截图(仅在 image 模式下)
        if mode == "image" and screenshot_paths:
            for screenshot_path in screenshot_paths:
                screenshot_file = Path(screenshot_path)
                
                if screenshot_file.exists():
                    try:
                        img_base64 = encode_image(str(screenshot_file))
                        
                        # 使用工具函数获取MIME类型
                        mime_type = get_image_mime_type(str(screenshot_file))
                        
                        # 添加截图说明文本
                        user_content.append({
                            "type": "text",
                            "text": f"\n[Target Screenshot: {screenshot_file.name}]"
                        })
                        
                        # 添加截图本体
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_base64}"
                            }
                        })
                        
                    except Exception as e:
                        print(f"Warning: Failed to encode screenshot {screenshot_path}: {e}")
        
        # 4. 添加资源信息
        if resources_info:
            resource_instruction = "\n## Available Resources\n"
            resource_instruction += "The following resources are available in your workspace:\n\n"
            
            for resource in resources_info:
                resource_type = resource.get("type", "")
                resource_path = resource.get("path", "")
                
                if mode == "text":
                    # text模式: 添加路径和描述(如果是图片)
                    resource_instruction += f"- `{resource_path}`"
                    if resource_type == "image" and resource.get("description"):
                        resource_instruction += f"\n  - Description: {resource['description']}"
                    resource_instruction += "\n"
                
                elif mode == "image":
                    # image模式: 添加路径
                    resource_instruction += f"- `{resource_path}`\n"
            
            user_content.append({
                "type": "text",
                "text": resource_instruction
            })
        
        # 5. image模式下,添加资源图片本体(与path绑定)
        if mode == "image" and workspace_path:
            
            for resource in resources_info:
                resource_type = resource.get("type", "")
                resource_path = resource.get("path", "")
                
                # 只处理图片类型
                if resource_type == "image" and resource_path:
                    full_path = workspace_path / resource_path
                    
                    if full_path.exists():
                        try:
                            img_base64 = encode_image(str(full_path))
                            
                            # 使用工具函数获取MIME类型
                            mime_type = get_image_mime_type(str(full_path))
                            
                            # 添加图片说明文本(绑定path)
                            user_content.append({
                                "type": "text",
                                "text": f"\n[Resource Image: {resource_path}]"
                            })
                            
                            # 添加图片本体
                            user_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{img_base64}"
                                }
                            })
                            
                        except Exception as e:
                            print(f"Warning: Failed to encode image {full_path}: {e}")
        
        # 构建 user message
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def construct_messages_for_edit_repair(self,
                                description: str,
                                src_code: List[Dict[str, str]],
                                src_screenshots: List[str],
                                dst_screenshots: List[str],
                                instruction_prompt: str,
                                mode: str = "text") -> List[Dict[str, Any]]:
        """
        根据不同模态组装 edit/repair 任务的 messages
    
        Args:
            description: 任务描述
            src_code: 源代码文件列表,格式为 [{"path": "...", "code": "..."}]
            src_screenshots: 源截图的完整路径列表
            dst_screenshots: 目标截图的完整路径列表
            instruction_prompt: 任务提示词
            mode: 模态类型,"text" 或 "image"
    
        Returns:
            List[Dict[str, Any]]: OpenAI 格式的 messages 列表
        """
        messages = []
        user_content = []
        
        # 1. 添加任务指令
        task_instruction = instruction_prompt + "\n" + f"## Task Description\n{description}\n"
        
        # 2. 添加源代码 (使用XML格式)
        task_instruction += "\n## Source Code\n"
        task_instruction += "The following is the current code that needs to be modified:\n\n"
        task_instruction += "<code_context>\n"
        
        for file_info in src_code:
            file_path = file_info.get("path", "")
            file_code = file_info.get("code", "")
            task_instruction += f'<file path="{file_path}">\n{file_code}\n</file>\n'
              
        task_instruction += "</code_context>\n"
        
        user_content.append({
            "type": "text",
            "text": task_instruction
        })
        
        # 3. 添加源截图(仅在 image 模式下)
        if mode == "image" and src_screenshots:
            user_content.append({
                "type": "text",
                "text": "\n## Current State Screenshots\nThe following screenshots show the current state:\n\n"
            })
            
            for screenshot_path in src_screenshots:
                screenshot_file = Path(screenshot_path)
                
                if screenshot_file.exists():
                    try:
                        img_base64 = encode_image(str(screenshot_file))
                        mime_type = get_image_mime_type(str(screenshot_file))
                        
                        user_content.append({
                            "type": "text",
                            "text": f"\n[Current Screenshot: {screenshot_file.name}]"
                        })
                        
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_base64}"
                            }
                        })
                        
                    except Exception as e:
                        print(f"Warning: Failed to encode screenshot {screenshot_path}: {e}")
    
        # 4. 添加目标截图(仅在 image 模式下)
        if mode == "image" and dst_screenshots:
            user_content.append({
                "type": "text",
                "text": "\n## Target State Screenshots\nThe following screenshots show the expected result:\n\n"
            })
            
            for screenshot_path in dst_screenshots:
                screenshot_file = Path(screenshot_path)
                
                if screenshot_file.exists():
                    try:
                        img_base64 = encode_image(str(screenshot_file))
                        mime_type = get_image_mime_type(str(screenshot_file))
                        
                        user_content.append({
                            "type": "text",
                            "text": f"\n[Target Screenshot: {screenshot_file.name}]"
                        })
                        
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_base64}"
                            }
                        })
                        
                    except Exception as e:
                        print(f"Warning: Failed to encode screenshot {screenshot_path}: {e}")
    
        messages.append({
            "role": "user",
            "content": user_content
        })
    
        return messages

    def parse_and_save_code(self, response: str, workspace_path: Path) -> List[Dict[str, str]]:
        """
        解析 LLM 生成的代码(使用 <file path="..."></file> 格式)并保存到 workspace
        
        Args:
            response: LLM 生成的响应内容
            workspace_path: 工作空间路径
        
        Returns:
            List[Dict[str, str]]: 保存的代码文件列表,格式为 [{"path": "...", "code": "..."}]
        """
        saved_files = []
        
        # 匹配 <file path="...">...</file> 格式
        pattern = r'<file\s+path=["\']([^"\']+)["\']>\s*(.*?)\s*</file>'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            for file_path_str, code in matches:
                # 清理路径和代码
                file_path_str = file_path_str.strip()
                code = code.strip()
                
                # 移除代码块标记(如果存在)
                code = re.sub(r'^```\w*\n?', '', code)
                code = re.sub(r'\n?```$', '', code)
                code = code.strip()
                
                # 构建完整路径
                file_path = workspace_path / file_path_str
                
                # 创建目录
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 保存文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                saved_files.append({"path": file_path_str, "code": code})
                print(f"Saved: {file_path_str}")
        
        
        return saved_files
   
    def run_generation_task(self, 
                           data_folder: Union[str, Path], 
                           mode: str = "text",
                           instruction_prompt: str = ""):
        """
        运行完整的 generation 任务流程
        
        Args:
            data_folder: generation 任务的数据文件夹路径
            mode: 模态类型,"text" 或 "image"
            instruction_prompt: 系统提示词
        
        """
        data_folder = Path(data_folder)
        workspace_path = data_folder / f"ans_{self.model_name}_{mode}"
        
        # 1. 加载 generation 任务数据
        description, screenshot_paths, resources_info = self.load_generation_data(data_folder)
        print(f"Target screenshots: {len(screenshot_paths)} items")
        print(f"Resources: {len(resources_info)} items")
        
        # 2. 创建 workspace (根据 resources_info 复制资源)
        self.create_workspace(data_folder, workspace_path, resources_info)
        print(f"Workspace created at: {workspace_path}")
        
        # 3. 组装 messages
        messages = self.construct_messages_for_generation(
            description=description,
            screenshot_paths=screenshot_paths,
            resources_info=resources_info,
            instruction_prompt=instruction_prompt,
            mode=mode,
            workspace_path=workspace_path
        )
        print(f"Messages constructed with mode: {mode}")
        
        # 4. 调用 LLM 生成代码
        response = self.chat(messages)
        print("LLM response received")
        
        # 5. 解析并保存代码到 workspace
        saved_files = self.parse_and_save_code(response, workspace_path)
        print(f"Saved {len(saved_files)} files")
        
        # 6. 截图
        screenshot_files = save_screenshots(str(workspace_path))
        print(f"Screenshots: {screenshot_files}")
        
        llm_log = {
            "ans_screenshot": screenshot_files,
            "workspace_path": str(workspace_path),
            "llm_input_messages": messages,
            "llm_response": response
        }
        
        with open(data_folder / f"{self.model_name}_{mode}_log.json", 'w', encoding='utf-8') as f:
            json.dump(llm_log, f, ensure_ascii=False, indent=4)
    
    def parse_and_apply_search_replace(self, response: str, src_code: List[Dict[str, str]], workspace_path: Path) -> List[Dict[str, str]]:
        """
        解析 LLM 生成的 search_replace 块并应用到源代码
    
        Args:
            response: LLM 生成的响应内容
            src_code: 源代码文件列表,格式为 [{"path": "...", "code": "..."}]
            workspace_path: 工作空间路径
    
        Returns:
            List[Dict[str, str]]: 修改后的代码文件列表,格式为 [{"path": "...", "code": "..."}]
        """
        # 解析 search_replace 块
        pattern = r'<search_replace\s+path=["\']([^"\']+)["\']>\s*<search>(.*?)</search>\s*<replace>(.*?)</replace>\s*</search_replace>'
        matches = re.findall(pattern, response, re.DOTALL)
        
        modified_files_list = []
        for file_path, search_text, replace_text in matches:
            search_stripped = search_text.strip()
            replace_stripped = replace_text.strip()
            
            # 检查 search 和 replace 是否相同
            if search_stripped == replace_stripped:
                print(f"Warning: search and replace are identical for path {file_path.strip()}")
                continue
            
            modified_files_list.append({
                "path": file_path.strip(),
                "search": search_stripped,
                "replace": replace_stripped,
            })
        
        if not modified_files_list:
            print("Warning: No valid search_replace blocks found in response")
            # 仍然保存原始代码到 workspace
            for file_info in src_code:
                file_path = workspace_path / file_info["path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(file_info["code"])
            return src_code
        
        # 使用工具函数应用 search_replace
        modified_code = apply_search_replace(src_code, modified_files_list)
        
        # 保存修改后的代码到 workspace
        for file_info in modified_code:
            file_path = workspace_path / file_info["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_info["code"])
            
            print(f"Saved: {file_info['path']}")
    
        return modified_code

    def run_edit_repair_task(self,
                        data_folder: Union[str, Path],
                        mode: str = "text",
                        instruction_prompt: str = ""):
        """
        运行完整的 edit/repair 任务流程
    
        Args:
            data_folder: edit/repair 任务的数据文件夹路径
            mode: 模态类型,"text" 或 "image"
            instruction_prompt: 系统提示词
    
        Returns:
            Dict[str, Any]: 包含任务执行结果的字典
        """
        data_folder = Path(data_folder)
        workspace_path = data_folder / f"ans_{self.model_name}_{mode}"
    
        # 1. 加载 edit/repair 任务数据
        description, src_code, src_screenshots, dst_screenshots, resources_info = self.load_edit_repair_data(data_folder)
        print(f"Source code files: {len(src_code)} items")
        print(f"Source screenshots: {len(src_screenshots)} items")
        print(f"Target screenshots: {len(dst_screenshots)} items")
    
        # 2. 创建 workspace 并复制源代码
        self.create_workspace(data_folder, workspace_path, resources_info)
        print(f"Workspace created at: {workspace_path}")
    
        # 3. 组装 messages
        messages = self.construct_messages_for_edit_repair(
            description=description,
            src_code=src_code,
            src_screenshots=src_screenshots,
            dst_screenshots=dst_screenshots,
            instruction_prompt=instruction_prompt,
            mode=mode
        )
        print(f"Messages constructed with mode: {mode}")
    
        # 4. 调用 LLM 生成修改指令
        response = self.chat(messages)
        print("LLM response received")
    
        # 5. 解析并应用 search_replace 到 workspace
        modified_files = self.parse_and_apply_search_replace(response, src_code, workspace_path)
        print(f"Modified {len(modified_files)} files")
    
        # 6. 截图
        screenshot_files = save_screenshots(str(workspace_path))
        print(f"Screenshots: {screenshot_files}")
    
        llm_log = {
            "ans_screenshot": screenshot_files,
            "workspace_path": str(workspace_path),
            "llm_input_messages": messages,
            "llm_response": response,
            "modified_files": modified_files
        }
    
        with open(data_folder / f"{self.model_name}_{mode}_log.json", 'w', encoding='utf-8') as f:
            json.dump(llm_log, f, ensure_ascii=False, indent=4)


