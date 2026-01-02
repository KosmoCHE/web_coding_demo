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
            "path": "文件相对路径",
            "code": "具体的代码内容"
        }
    ],
    "dst_code": [
        {
            "path": "文件相对路径",
            "code": "具体的代码内容"
        }
    ],
    "src_screenshot": ["图片文件名1","图片文件名2"],
    "dst_screenshot": ["图片文件名1","图片文件名2"],
    "resources": [
        {
            "type": "image/other",
            "path": "文件相对路径",
            "description": "资源文件的说明(only for image)",
        }
    ],
    "label_modified_files": [
        {
            "path": "文件相对路径",
            "search": "被修改/删除的代码片段",
            "replace": "修改后的代码片段"
        }
    ]
}
```

folder structure:
```
.
└── generation
    ├── 1009769_www.kccworld.co.kr_english_
    │   ├── dst
    │   │   ├── index.html
    │   │   ├── resources
    │   │   │   ├── common.css
    │   │   │   ├── content.css
        │   │   └── img_wa_mark.png
        │   └── screenshot_0.jpg
        └── info.json

```