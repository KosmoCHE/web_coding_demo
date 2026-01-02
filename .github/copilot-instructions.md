# Web Coding Dataset Synthesis Project

## Project Overview
This project generates synthetic web development datasets for training LLMs on three task types: **generation** (creating webpages from descriptions), **edit** (modifying existing code), and **repair** (fixing defects). Data flows from source webpages → LLM-based transformations → structured JSON outputs with code, screenshots, and metadata.

## Architecture & Data Flow

### Core Components
- **`synthetic/`**: Task synthesizers that use LLMs to generate training data
  - `synthesizer.py`: Base class with XML parsing (`<search_replace>` blocks) and code transformation logic
  - `edit.py`: Forward evolution - modifies clean code to create edit tasks (5 task types: Function/Element/Style Modification, Add/Remove Element)
  - `repair.py`: Reverse defect injection - introduces bugs into clean code (12 defect types: Occlusion, Crowding, Text Overlap, etc.)
  - `transform_renderbench.py`: Converts raw webpage data to standardized format with LLM-generated descriptions
- **`mllm/`**: Multi-modal LLM abstraction layer
  - `base.py`: Abstract `MLLMChat` class with image encoding and data loading utilities
  - `openai_chat.py`: OpenAI API implementation
- **`utils/`**: Shared utilities
  - `config.py`: File extension constants (`CODE_EXTENSIONS`, `IMAGE_EXTENSIONS`)
  - `utils.py`: Image encoding (SVG→PNG conversion), retry logic, Playwright screenshot capture

### Data Structure
All tasks follow this `info.json` schema in `data/data_demo_renderbench/`:
```json
{
  "task": "generation|edit|repair",
  "task_type": ["subtask1", "subtask2"],
  "description": "Natural language task instruction",
  "src_code": [{"path": "...", "code": "..."}],  // Edit/repair only
  "dst_code": [{"path": "...", "code": "..."}],
  "src_screenshot": ["..."],  // Edit/repair only
  "dst_screenshot": ["..."],
  "resources": [{"type": "image|other", "path": "...", "description": "..."}],
  "label_modified_files": [{"path": "...", "search": "...", "replace": "..."}]
}
```

## Critical Workflows

