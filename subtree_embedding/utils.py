from typing import Dict, Any, List, Optional, Tuple
import re

class NodeTypeClassifier:
    """
    Node type classifier
    """
    
    def classify(self, node) -> str:
        """
        Classify the node type based on element_type
        """
        if isinstance(node, dict):
            metadata = node.get('metadata', {})
            element_type = metadata.get('element_type')
            
            if element_type:
                # 直接映射element_type到标准类型
                type_mapping = {
                    'paragraph': 'text',
                    'figure': 'image', 
                    'table': 'table',
                    'page_header': 'text',
                    'page_footer': 'text',
                    'heading': 'heading',
                    'list': 'list'
                }
                return type_mapping.get(element_type, 'text')
            
            # 如果没有element_type，返回默认值
            return 'text'
        
        return 'unknown'

class StructuralPromptGenerator:
    """
    Structural prompt generator
    """
    
    def generate_prompt(self, node, depth: int) -> str:
        """
        Generate the structural prompt
        """
        
        # Handle dictionary-based nodes (our DOM structure)
        if isinstance(node, dict):
            metadata = node.get('metadata', {})
            
            # 使用page_number而不是page_id（我们的字段名）
            page_num = metadata.get('page_number')
            if page_num is not None:
                page_num += 1  # 转换为1-based索引
            
            # 使用element_type（我们的字段名）
            element_type = metadata.get('element_type', 'unknown')
            
            # 构建提示
            prompt_parts = []
            
            # 添加页面信息
            if page_num is not None:
                prompt_parts.append(f"[Page {page_num}]")
            
            # 添加深度信息
            if depth > 0:
                prompt_parts.append(f"[Depth {depth}]")
            
            # 添加节点类型
            if element_type != 'unknown':
                prompt_parts.append(f"[{element_type.capitalize()}]")
            
            # 添加bbox位置信息（我们的特有信息）
            bbox = metadata.get('bbox')
            if bbox and len(bbox) == 4:
                prompt_parts.append(f"[Pos: {bbox[0]:.0f},{bbox[1]:.0f}]")
            
            # 添加标题层级信息（如果是标题）
            heading_level = metadata.get('heading_level')
            if heading_level:
                prompt_parts.append(f"[H{heading_level}]")
            
            # 添加父级上下文（如果有）
            if 'parent_context' in metadata:
                prompt_parts.append(f"under {metadata['parent_context']}")
            
            return " ".join(prompt_parts) + ":"
        
        # Handle object-based nodes (original format)  
        else:
            node_type = getattr(node, 'node_type', 'unknown')
            page_num = getattr(node, 'page_id', 'unknown')
            metadata = getattr(node, 'metadata', {})
            
            # build the prompt
            prompt_parts = []
            
            # add the page number (if any)
            if page_num and page_num != 'unknown':
                prompt_parts.append(f"[Page {page_num}]")
            
            # add the depth (if any)
            if depth > 0:
                prompt_parts.append(f"[Depth {depth}]")
            
            # add the node type
            if node_type and node_type != 'unknown':
                prompt_parts.append(f"[{node_type.title()}]")
            
            # add the parent context (if any)
            if 'parent_context' in metadata:
                prompt_parts.append(f"under {metadata['parent_context']}")
            
            return " ".join(prompt_parts) + ":"


