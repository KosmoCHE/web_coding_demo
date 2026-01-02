## 12.25
### 数据生成
1. 通过注入的任务数量来控制难度

### 模型测试输入输出格式
#### Generation任务
- 输入（文本）
  1. Description: 任务描述
  2. Resources：相对路径 + 描述（仅针对大模型支持的图片）
- 输入（图片）
  1. Description: 任务描述
  2. dst screenshot
  3. Resources：相对路径 + 描述（仅针对大模型支持的图片）
- 输出（完整代码）
```
<file path="path/to/generated_file.html">
</file>
```

#### Edit/Repair任务
- 输入（文本）
  1. Description: 任务描述
  2. Src code: 有缺陷的代码
- 输入（图片）
  1. Description: 任务描述
  2. Src code: 有缺陷的代码
  3. Src screenshot
  4. dst screenshot
- 输出（代码片段）
```
<search_replace path="path/to/modified_file.html">
<search>
code in src to be replaced
</search>
<replace>
code after replacement
</replace>
</search_replace>
```
### Eval 指标
1. Generation任务
text mode: 仅检查视觉美观度与资源利用情况（人工评估）
image mode: 评估截图的视觉美观度 + 资源利用情况（人工评估）
2. Edit/Repair任务
检查任务完成情况
评分标准：
level 0: 任务未完成，代码无变化或与需求无关
level 1: 定位到问题，但修改不完整或有误
level 2: 定位到问题，修改基本正确，但有细节错误
level 3: 完全正确，代码符合需求且无错误
### Problem & Conclusion
Problem:
1. text mode，generation如何评估，没有参考截图，只能通过美观度和功能性来评估，比较困难
2. edit/repair任务，模型输出的代码片段，有时无法直接定位到对应的文件（fail）

Conclusion:
1. image mode下，repair/edit任务，模型表现更好，减少了  level2 的情况

### TODO
1. 直接通过llm judge 任务edit/repair的完成情况
2. 提高edit任务难度