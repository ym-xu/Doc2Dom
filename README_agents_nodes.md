# 检索Nodes内容格式说明

## 概述

本系统会为每个检索问题保存检索到的top-k nodes的原始内容，供agents系统使用。这些内容保存在 `./retrieved_nodes_for_agents/` 目录中。

## 文件命名规则

```
{doc_id}_{question_hash}.json
```

- `doc_id`: 文档ID（去掉.pdf后缀）
- `question_hash`: 问题的MD5 hash前8位

例如：`PH_2016.06.08_Economy-Final_a1b2c3d4.json`

## 数据结构

### 根层级

```json
{
  "doc_id": "文档ID",
  "question": "检索问题",
  "retrieval_timestamp": "检索时间戳",
  "total_nodes": "检索到的节点总数",
  "nodes_content": [节点内容数组]
}
```

### 节点内容结构

每个节点包含以下信息：

#### 基本信息
- `node_id`: 节点唯一标识符
- `rank`: 在检索结果中的排名（1-based）
- `similarity_score`: 相似度分数（0-1）
- `page_number`: 页面编号（0-based，需要+1转换为实际页码）
- `element_type`: 元素类型（text/figure/table/heading等）

#### 内容信息
- `text_content`: 节点的原始文本内容
- `bbox`: 元素在页面中的位置框 [x1, y1, x2, y2]
- `dom_path`: DOM路径信息

#### 结构信息
- `heading_level`: 标题级别（如果是heading）
- `is_chapter_title`: 是否为章节标题
- `parent_chapter`: 所属章节
- `depth`: DOM树中的深度

#### 类型特定信息

**图片信息** (`image_info`)：
- `ai_description`: AI生成的图片描述
- `description_method`: 描述生成方法
- `image_extracted`: 是否成功提取图片

**表格信息** (`table_info`)：
- `table_image_extracted`: 是否成功提取表格图片
- `row_count`: 行数
- `col_count`: 列数

## 使用示例

### Python读取示例

```python
import json

def load_retrieved_nodes(filepath):
    """加载检索到的nodes内容"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_text_content(nodes_data):
    """提取所有节点的文本内容"""
    texts = []
    for node in nodes_data['nodes_content']:
        content = f"Page {node['page_number'] + 1}: {node['text_content']}"
        
        # 如果是图片，添加AI描述
        if node['image_info'] and node['image_info']['ai_description']:
            content += f" [图片描述: {node['image_info']['ai_description']}]"
        
        texts.append(content)
    
    return texts

# 使用示例
data = load_retrieved_nodes('./retrieved_nodes_for_agents/example_doc_a1b2c3d4.json')
print(f"问题: {data['question']}")
print(f"检索到 {data['total_nodes']} 个相关节点")

for node in data['nodes_content'][:3]:  # 显示前3个
    print(f"排名 {node['rank']}: {node['text_content'][:100]}...")
```

### 按类型筛选节点

```python
def filter_nodes_by_type(nodes_data, element_type):
    """按元素类型筛选节点"""
    return [node for node in nodes_data['nodes_content'] 
            if node['element_type'] == element_type]

# 获取所有图片节点
figures = filter_nodes_by_type(data, 'figure')
for fig in figures:
    print(f"图片: {fig['text_content']}")
    if fig['image_info']:
        print(f"  描述: {fig['image_info']['ai_description']}")

# 获取所有表格节点
tables = filter_nodes_by_type(data, 'table')
for table in tables:
    print(f"表格: {table['text_content']}")
    if table['table_info']:
        print(f"  规模: {table['table_info']['row_count']}行 x {table['table_info']['col_count']}列")
```

### 构建上下文

```python
def build_context_for_agent(nodes_data, max_length=2000):
    """为agent构建上下文"""
    context_parts = []
    current_length = 0
    
    question = nodes_data['question']
    context_parts.append(f"问题: {question}")
    current_length += len(question)
    
    context_parts.append("\n检索到的相关内容:")
    
    for node in nodes_data['nodes_content']:
        node_text = f"\n[页面{node['page_number'] + 1}, 排名{node['rank']}, 相似度{node['similarity_score']:.3f}]\n"
        node_text += node['text_content']
        
        # 添加图片描述
        if node['image_info'] and node['image_info']['ai_description']:
            node_text += f"\n[图片描述: {node['image_info']['ai_description']}]"
        
        if current_length + len(node_text) > max_length:
            break
            
        context_parts.append(node_text)
        current_length += len(node_text)
    
    return "\n".join(context_parts)
```

## 注意事项

1. **页面编号**: 保存的`page_number`是0-based，实际页码需要+1
2. **文本截断**: `text_content`可能是截断的预览，完整内容需要从原始DOM中获取
3. **图片描述**: AI描述的质量取决于原始处理过程，可能包含"Skipped"或"Failed"等状态信息
4. **表格内容**: 目前主要保存表格的元信息，具体表格数据需要进一步处理

## 目录结构

```
./retrieved_nodes_for_agents/
├── doc1_hash1.json
├── doc1_hash2.json
├── doc2_hash1.json
└── ...
```

每个文件对应一个文档的一个问题的检索结果。