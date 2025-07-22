# JSON到DOM转换器使用指南

## 功能概述

这个工具将从PDF解析得到的JSON文件转换为结构化的DOM树，支持：

1. **基于parent_chapter的层次结构**：按语义相关性组织内容
2. **智能文本合并**：合并疑似列表项的短句子
3. **图像提取与描述**：从PDF提取图片并使用AI生成描述
4. **表格处理**：提取表格内容和图片

## 核心特性

### 1. DOM树结构

```
Document (根节点)
├── Chapter 1 (is_chapter_title=true)
│   ├── Paragraph
│   ├── Figure (带图片和AI描述)
│   └── Table (带单元格和图片)
├── Chapter 2
│   └── ...
└── 其他顶级元素
```

### 2. 文本合并逻辑

- **触发条件**：短句子（平均<30字符）且总长度<128字符
- **空间连续性**：基于坐标判断元素是否连续
- **列表识别**：自动识别以•、-、*或数字开头的列表项

### 3. 图像处理流程

1. 从JSON中读取`outline`坐标
2. 使用PyMuPDF从PDF提取图片
3. 调用GPT生成图像描述
4. 保存到 `data/dom/MMLongBench-Doc/文档名/图片名.png`

### 4. 表格处理

- 解析`cells`字典结构（"row_col": {"text": "内容"}）
- 构建HTML表格DOM结构
- 同时提取表格区域图片

## 使用方法

### 基本使用

```python
from json_to_dom_processor import process_document

# 处理单个文档
dom_tree = process_document(
    json_path="data/dict/MMLongBench-Doc/document.json",
    pdf_path="data/pdf/document.pdf",  # 可选，用于图像提取
    max_merge_chars=128  # 文本合并字符限制
)

# 保存结果
import json
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(dom_tree.to_json_dict(), f, indent=2, ensure_ascii=False)
```

### 启用AI图像描述

```python
# 使用GPT-4V
dom_tree = process_document(
    json_path="document.json",
    pdf_path="document.pdf",
    openai_api_key="your-openai-api-key",
    prefer_model="gpt"
)

# 使用Qwen2VL
dom_tree = process_document(
    json_path="document.json", 
    pdf_path="document.pdf",
    qwen2vl_model="Qwen/Qwen2-VL-7B-Instruct",
    prefer_model="qwen2vl"
)
```

### 批量处理

```python
from json_to_dom_processor import batch_process_documents

results = batch_process_documents(
    json_dir="data/dict/MMLongBench-Doc",
    pdf_dir="data/pdf",  # 可选
    openai_api_key="your-api-key",
    prefer_model="gpt",
    max_merge_chars=128
)

# results是一个字典: {文档名: DOMNode}
for doc_name, dom_tree in results.items():
    print(f"处理完成: {doc_name}, 子节点数: {len(dom_tree.children)}")
```

## 配置参数

### 核心参数

- **max_merge_chars** (int, 默认128): 文本合并的最大字符数
- **prefer_model** (str, "gpt"/"qwen2vl"): 优先使用的AI模型
- **openai_api_key** (str): OpenAI API密钥
- **qwen2vl_model** (str): Qwen2VL模型名称

### 目录结构

```
data/
├── dict/MMLongBench-Doc/           # 输入JSON文件
│   ├── document1.json
│   └── document2.json
├── pdf/                            # 输入PDF文件（可选）
│   ├── document1.pdf
│   └── document2.pdf
└── dom/MMLongBench-Doc/           # 输出目录
    ├── document1/                  # 每个文档一个目录
    │   ├── page_1_figure_2.png     # 提取的图片
    │   └── page_2_table_5.png
    └── document1.json              # DOM树JSON
```

## 输出格式

### DOM节点结构

```json
{
  "tag": "figure",
  "text": "图表标题",
  "src": "document1/page_1_figure_2.png",
  "metadata": {
    "depth": 2,
    "global_id": "doc_document1_42",
    "original_index": 15,
    "parent_chapter": 8,
    "element_type": "figure",
    "ai_description": "这是一个显示销售趋势的柱状图...",
    "bbox": [100, 200, 400, 350]
  },
  "children": [
    {
      "tag": "img",
      "src": "document1/page_1_figure_2.png",
      "alt": "AI生成的图像描述"
    }
  ]
}
```

### 图像描述示例

AI生成的图像描述包含：
1. 图像类型识别（图表、照片、示意图等）
2. 关键视觉元素描述
3. 图像中的文本内容
4. 在文档中的作用和目的
5. 技术细节（如果是图表）

## 调试和错误处理

### 常见问题

1. **图像提取失败**
   - 检查PDF文件路径是否正确
   - 确认PyMuPDF依赖已安装

2. **AI描述失败**
   - 检查API密钥是否有效
   - 确认网络连接和API端点

3. **文本合并异常**
   - 调整`max_merge_chars`参数
   - 检查JSON中的`outline`坐标数据

### 日志输出

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

dom_tree = process_document(json_path, pdf_path)
```

## 依赖项

```bash
pip install PyMuPDF Pillow openai requests
```

## API配置示例

### OpenAI GPT-4V

```python
openai_api_key = "sk-your-openai-api-key"
```

### Qwen2VL (本地部署)

```python
qwen2vl_model = "Qwen/Qwen2-VL-7B-Instruct"
qwen2vl_device = 0  # CUDA设备编号
```

## 性能优化建议

1. **批量处理**：使用`batch_process_documents`提高效率
2. **并发控制**：大量文档时考虑进程池
3. **缓存机制**：已提取的图片可以缓存重用
4. **API限制**：注意AI服务的调用频率限制

## 扩展功能

### 自定义图像描述提示词

```python
class CustomImageService(ImageDescriptionService):
    def get_image_description_prompt(self, page_context=""):
        return "自定义的提示词模板..."

# 使用自定义服务
processor = JSONToDOMProcessor()
processor.image_service = CustomImageService(openai_api_key)
```

### 自定义文本合并逻辑

```python
class CustomProcessor(JSONToDOMProcessor):
    def _should_merge_elements(self, elements):
        # 自定义合并逻辑
        return custom_merge_logic(elements)
```