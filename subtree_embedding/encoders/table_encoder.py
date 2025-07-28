import numpy as np
from typing import List

class TableEncoder:
    """
    简化的Table encoder - 专门处理我们的DOM结构中的表格
    """
    
    def __init__(self, shared_model, shared_processor, target_dim: int = 768, device: str = "cpu"):
        """
        TableEncoder现在接收共享模型，不再独立加载TextEncoder
        """
        from .text_encoder import TextEncoder
        self.text_encoder = TextEncoder(shared_model, shared_processor, target_dim, device)
        self.target_dim = target_dim
        self.device = device
    
    def encode(self, node, structure_prompt: str = "") -> np.ndarray:
        """
        编码表格节点 - 接收结构信息作为上下文的一部分
        """
        try:
            # 提取表格内容
            table_content = self._extract_table_content(node)
            
            # 构建增强的文本上下文（包含结构信息和AI描述）
            enhanced_text = self._build_enhanced_context(table_content, structure_prompt, node)
            
            if not enhanced_text.strip():
                return np.zeros(self.target_dim)
            
            return self.text_encoder.encode(enhanced_text)
            
        except Exception as e:
            print(f"Error in table encoder: {e}")
            return np.zeros(self.target_dim)
    
    def _extract_table_content(self, node) -> str:
        """
        从DOM节点中提取表格内容
        """
        if not isinstance(node, dict):
            return ""
        
        # 收集所有行的内容
        rows = []
        children = node.get('children', [])
        
        for child in children:
            if isinstance(child, dict) and child.get('tag') == 'tr':
                # 提取行中的所有单元格
                cells = self._extract_row_cells(child)
                if cells:
                    rows.append(cells)
        
        if not rows:
            # 如果没有找到行结构，返回节点的text内容
            return node.get('text', '') or 'Empty table'
        
        # 将表格转换为简单的文本格式
        return self._format_table_as_text(rows)
    
    def _extract_row_cells(self, row_node) -> List[str]:
        """
        从tr节点中提取所有td/th单元格的文本
        """
        cells = []
        children = row_node.get('children', [])
        
        for cell in children:
            if isinstance(cell, dict) and cell.get('tag') in ['td', 'th']:
                cell_text = cell.get('text', '').strip()
                cells.append(cell_text)
        
        return cells
    
    def _format_table_as_text(self, rows: List[List[str]]) -> str:
        """
        将表格行转换为文本格式
        """
        if not rows:
            return ""
        
        # 简单的文本格式：每行用 | 分隔单元格
        formatted_rows = []
        for row in rows:
            # 清理单元格内容（移除多余的换行和空格）
            cleaned_cells = [cell.replace('\n', ' ').strip() for cell in row]
            formatted_row = " | ".join(cleaned_cells)
            formatted_rows.append(formatted_row)
        
        return "\n".join(formatted_rows)
    
    def _build_enhanced_context(self, table_content: str, structure_prompt: str, node) -> str:
        """
        构建包含结构信息和AI描述的增强文本上下文
        """
        context_parts = []
        
        # 1. 添加结构信息（如果提供）
        if structure_prompt.strip():
            context_parts.append(structure_prompt.strip())
        
        # 2. 添加表格内容
        if table_content.strip():
            context_parts.append(f"Table: {table_content}")
        
        # 3. 添加AI描述（如果有效）
        if isinstance(node, dict):
            metadata = node.get('metadata', {})
            ai_desc = metadata.get('ai_description', '')
            if (ai_desc and 
                ai_desc != 'Skipped (disabled or too small)' and 
                not ai_desc.startswith('Failed') and
                len(ai_desc.strip()) > 10):
                context_parts.append(f"Description: {ai_desc}")
        
        return ". ".join(context_parts) if context_parts else ""