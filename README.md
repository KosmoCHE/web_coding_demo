# Web Coding Dataset Demo

## data folder
info.json: Metadata for each sample in the dataset.
generation task only includes `dst_code` and `dst_screenshot`.
```json
{
    "task": "generation/edit/repair",
    "task_type": ["subtask1", "subtask2"],
    "description":"具体要生成/编辑/修复的内容说明(外观/交互)",
    "src_code": [
        {
            "language": "html/css/js",
            "directory": "文件相对路径",
            "code": "具体的代码内容"
        }
    ],
    "src_screenshot": ["图片文件名1","图片文件名2"],
    "dst_code": [
        {
            "language": "html/css/js",
            "directory": "文件相对路径",
            "code": "具体的代码内容"
        }
    ],
    "dst_screenshot": ["图片文件名1","图片文件名2"]
}
```