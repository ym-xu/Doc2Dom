#!/usr/bin/env python3
"""
DOM树结构交互式可视化工具
真正的树形展示，支持点击展开、图片预览、表格预览、文本展开
"""

import json
import sys
import argparse
import os
import webbrowser
from typing import Dict, Any, List, Optional
from collections import defaultdict

class DOMNode:
    """简化的DOM节点类用于可视化"""
    def __init__(self, data: Dict[str, Any]):
        self.tag = data.get('tag', 'unknown')
        self.text = data.get('text', '')
        self.metadata = data.get('metadata', {})
        self.attrs = {k: v for k, v in data.items() if k not in ['tag', 'text', 'metadata', 'children']}
        self.children = [DOMNode(child) for child in data.get('children', [])]
        self.depth = self.metadata.get('depth', 0)
        self.global_id = self.metadata.get('global_id', '')
        self.heading_level = self.metadata.get('heading_level', None)

class DOMInteractiveTree:
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.root = None
        self.load_tree()
        
    def load_tree(self):
        """加载DOM树"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.root = DOMNode(data)
            print(f"✅ 成功加载DOM树: {self.json_file}")
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            sys.exit(1)
    
    def generate_html(self, output_file: str = "interactive_tree.html"):
        """生成交互式HTML页面"""
        # 收集统计信息
        stats = self._collect_stats(self.root)
        
        # 生成DOM树JSON数据
        tree_data = self._node_to_dict(self.root)
        
        # 生成HTML内容
        html_content = self._create_html_page(stats, tree_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML可视化页面已生成: {output_file}")
        return output_file
    
    def _node_to_dict(self, node: DOMNode) -> Dict[str, Any]:
        """将DOM节点转换为字典"""
        icon = self._get_node_icon(node)
        node_class = self._get_node_class(node)
        
        # 获取图片和表格路径
        image_src = self._get_image_src(node)
        table_src = self._get_table_src(node)
        
        # 创建节点数据
        node_data = {
            'id': node.global_id or f"node_{id(node)}",
            'tag': node.tag,
            'text': node.text[:100] + "..." if len(node.text) > 100 else node.text,
            'full_text': node.text,
            'depth': node.depth,
            'icon': icon,
            'class': node_class,
            'image_src': image_src,
            'table_src': table_src,
            'metadata': {
                'heading_level': node.heading_level,
                'merged_count': node.metadata.get('merged_count', None),
                'element_type': node.metadata.get('element_type', ''),
                'page_number': node.metadata.get('page_number', 0),
                'bbox': node.metadata.get('bbox', [])
            },
            'children': [self._node_to_dict(child) for child in node.children]
        }
        
        return node_data
    
    def _get_node_icon(self, node: DOMNode) -> str:
        """获取节点图标"""
        if node.tag == 'document':
            return '📄'
        elif node.tag.startswith('h'):
            return '📖'
        elif node.tag == 'p':
            return '📝'
        elif node.tag == 'ul':
            return '📋'
        elif node.tag == 'figure':
            return '🖼️'
        elif node.tag == 'table':
            return '📊'
        elif node.tag == 'img':
            return '🖼️'
        elif node.tag in ['header', 'footer']:
            return '🏷️'
        else:
            return '📄'
    
    def _get_node_class(self, node: DOMNode) -> str:
        """获取节点CSS类"""
        classes = ['node']
        
        if node.tag.startswith('h'):
            classes.append('heading')
            classes.append(f'heading-{node.heading_level}')
        elif node.tag == 'ul':
            classes.append('merged-list')
        elif node.tag == 'figure':
            classes.append('image-node')
        elif node.tag == 'table':
            classes.append('table-node')
        elif node.tag == 'p':
            classes.append('paragraph')
        
        return ' '.join(classes)
    
    def _get_image_src(self, node: DOMNode) -> str:
        """获取图片路径"""
        if node.tag == 'figure':
            # 查找是否有图片文件
            document_name = node.metadata.get('document_name', 'welcome-to-nus.pdf')
            page_number = node.metadata.get('page_number', 0)
            global_id = node.global_id
            
            # 尝试不同的图片格式
            possible_paths = [
                f"./data/dom/MMLongBench-Doc/{document_name.replace('.pdf', '')}/page_{page_number}_figure_{global_id.split('_')[-1]}.png",
                f"./data/dom/MMLongBench-Doc/{document_name.replace('.pdf', '')}/page_{page_number}_figure_{global_id}.png",
                f"data/images/{document_name.replace('.pdf', '')}/page_{page_number}_figure.png"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        return ''
    
    def _get_table_src(self, node: DOMNode) -> str:
        """获取表格图片路径"""
        if node.tag == 'table':
            # 查找是否有表格图片文件
            document_name = node.metadata.get('document_name', 'welcome-to-nus.pdf')
            page_number = node.metadata.get('page_number', 0)
            global_id = node.global_id
            
            # 尝试不同的表格图片格式
            possible_paths = [
                f"./data/dom/MMLongBench-Doc/{document_name.replace('.pdf', '')}/page_{page_number}_table_{global_id.split('_')[-1]}.png",
                f"./data/dom/MMLongBench-Doc/{document_name.replace('.pdf', '')}/page_{page_number}_table_{global_id}.png",
                f"data/tables/{document_name.replace('.pdf', '')}/page_{page_number}_table.png"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        return ''
    
    def _collect_stats(self, node: DOMNode, stats: Optional[Dict] = None) -> Dict:
        """收集树的统计信息"""
        if stats is None:
            stats = {
                'total_nodes': 0,
                'max_depth': 0,
                'headings': defaultdict(int),
                'merged_nodes': 0,
                'image_nodes': 0,
                'table_nodes': 0,
                'depth_distribution': defaultdict(int)
            }
        
        stats['total_nodes'] += 1
        stats['max_depth'] = max(stats['max_depth'], node.depth)
        stats['depth_distribution'][node.depth] += 1
        
        if node.tag.startswith('h'):
            stats['headings'][node.tag] += 1
        
        if 'merged_count' in node.metadata:
            stats['merged_nodes'] += 1
        
        if node.tag == 'figure':
            stats['image_nodes'] += 1
        
        if node.tag == 'table':
            stats['table_nodes'] += 1
        
        for child in node.children:
            self._collect_stats(child, stats)
        
        return stats
    
    def _create_html_page(self, stats: Dict, tree_data: Dict) -> str:
        """创建简洁的DOM树HTML页面"""
        tree_data_json = json.dumps(tree_data, ensure_ascii=False, indent=2)
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DOM Tree Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            background: #ffffff;
            color: #333;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: none;
            background: white;
        }}
        
        .file-selector {{
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }}
        
        .file-selector label {{
            font-weight: bold;
            margin-right: 10px;
        }}
        
        .file-selector input[type="file"] {{
            margin-right: 10px;
        }}
        
        .file-selector button {{
            background: #007bff;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
        }}
        
        .file-selector button:hover {{
            background: #0056b3;
        }}
        
        .controls {{
            margin-bottom: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }}
        
        .controls button {{
            background: #6c757d;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 3px;
            cursor: pointer;
            margin: 2px;
            font-size: 12px;
        }}
        
        .controls button:hover {{
            background: #5a6268;
        }}
        
        .controls button.active {{
            background: #007bff;
        }}
        
        .tree {{
            font-size: 14px;
        }}
        
        .tree-node {{
            margin: 2px 0;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
            transition: background-color 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            position: relative;
        }}
        
        .tree-node:hover {{
            background: rgba(0, 123, 255, 0.1);
        }}
        
        .tree-node.selected {{
            background: rgba(0, 123, 255, 0.2);
        }}
        
        .tree-node.clickable {{
            border-left: 3px solid #28a745;
        }}
        
        .tree-node.clickable:hover {{
            background: rgba(40, 167, 69, 0.1);
        }}
        
        .toggle {{
            width: 16px;
            text-align: center;
            color: #666;
            font-size: 12px;
            user-select: none;
        }}
        
        .toggle:hover {{
            color: #333;
        }}
        
        .node-icon {{
            font-size: 16px;
            min-width: 20px;
        }}
        
        .node-tag {{
            font-weight: bold;
            color: #0056b3;
            min-width: 80px;
        }}
        
        .node-text {{
            color: #666;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            flex: 1;
        }}
        
        .node-badge {{
            background: #f8f9fa;
            color: #6c757d;
            padding: 2px 6px;
            border-radius: 8px;
            font-size: 11px;
            margin-left: auto;
        }}
        
        .content-indicator {{
            color: #28a745;
            font-size: 12px;
            margin-left: 4px;
        }}
        
        .children {{
            margin-left: 24px;
            border-left: 1px solid #dee2e6;
            padding-left: 12px;
            margin-top: 2px;
        }}
        
        .heading .node-tag {{
            color: #dc3545;
        }}
        
        .merged-list .node-tag {{
            color: #28a745;
        }}
        
        .image-node .node-tag {{
            color: #fd7e14;
        }}
        
        .table-node .node-tag {{
            color: #6f42c1;
        }}
        
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
        }}
        
        .modal-content {{
            background-color: #ffffff;
            margin: 5% auto;
            padding: 30px;
            border-radius: 8px;
            width: 90%;
            max-width: 900px;
            max-height: 80vh;
            overflow-y: auto;
            position: relative;
        }}
        
        .close {{
            color: #aaa;
            float: right;
            font-size: 24px;
            font-weight: bold;
            position: absolute;
            right: 20px;
            top: 15px;
            cursor: pointer;
        }}
        
        .close:hover {{
            color: #000;
        }}
        
        .preview-image {{
            max-width: 100%;
            max-height: 400px;
            object-fit: contain;
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        
        .text-content {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            max-height: 300px;
            overflow-y: auto;
            border-left: 3px solid #007bff;
            font-family: 'Georgia', serif;
            line-height: 1.6;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="file-selector">
            <label for="jsonFile">Select JSON file:</label>
            <input type="file" id="jsonFile" accept=".json">
        </div>
        
        <div class="controls" id="controls" style="display: none;">
            <strong>Expand:</strong>
            <button onclick="expandToDepth(1)">Level 1</button>
            <button onclick="expandToDepth(2)">Level 2</button>
            <button onclick="expandToDepth(3)">Level 3</button>
            <button onclick="expandAll()" class="active">All</button>
            <button onclick="collapseAll()">Collapse</button>
        </div>
        
        <div class="tree" id="treeContainer">
            <!-- Tree structure will be generated here -->
        </div>
    </div>
    
    <!-- Content preview modal -->
    <div id="contentModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <div id="modalContent">
                <!-- Modal content will be generated here -->
            </div>
        </div>
    </div>
    
    <script>
        // Global variables
        let treeData = null;
        let expandedNodes = new Set();
        let selectedNode = null;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            setupEventListeners();
        }});
        
        function setupEventListeners() {{
            const fileInput = document.getElementById('jsonFile');
            if (fileInput) {{
                fileInput.addEventListener('change', handleFileSelect);
            }}
        }}
        
        function handleFileSelect(event) {{
            const file = event.target.files[0];
            if (file) {{
                const reader = new FileReader();
                reader.onload = function(e) {{
                    try {{
                        treeData = JSON.parse(e.target.result);
                        initializeTree();
                    }} catch (error) {{
                        alert('Error parsing JSON file: ' + error.message);
                    }}
                }};
                reader.readAsText(file);
            }}
        }}
        
        
        function initializeTree() {{
            const controls = document.getElementById('controls');
            if (controls) {{
                controls.style.display = 'block';
            }}
            renderTree();
            expandAll(); // Default expand all nodes
        }}
        
        function renderTree() {{
            if (!treeData) return;
            
            const container = document.getElementById('treeContainer');
            container.innerHTML = renderNode(treeData, 0);
        }}
        
        function renderNode(node, depth) {{
            if (!node) {{
                return '';
            }}
            
            const hasChildren = node.children && node.children.length > 0;
            const isExpanded = expandedNodes.has(node.id);
            const isClickable = isContentNode(node);
            const canToggle = node.tag !== 'img';
            
            const nodeId = node.id || `node_${{Math.random().toString(36).substr(2, 9)}}`;
            const nodeClass = node.class || '';
            const icon = node.icon || '';
            const text = node.text || '';
            const displayText = text.length > 100 ? text.substring(0, 100) + '...' : text;
            
            let html = `
                <div class="tree-node ${{nodeClass}} ${{isClickable ? 'clickable' : ''}}" 
                     data-id="${{nodeId}}" onclick="selectNode('${{nodeId}}', event)">
                    <span class="toggle" onclick="${{canToggle ? `toggleNode('${{nodeId}}', event)` : ''}}" 
                          style="${{hasChildren && canToggle ? 'cursor: pointer;' : hasChildren ? 'cursor: default; color: #ccc;' : 'visibility: hidden;'}}">${{isExpanded ? '▼' : '▶'}}</span>
                    <span class="node-icon">${{icon}}</span>
                    <span class="node-tag">${{node.tag}}</span>
                    <span class="node-text">${{displayText}}</span>
                    ${{getNodeBadge(node)}}
                    ${{getContentIndicator(node)}}
                </div>
            `;
            
            if (hasChildren && isExpanded) {{
                html += '<div class="children">';
                for (const child of node.children) {{
                    html += renderNode(child, depth + 1);
                }}
                html += '</div>';
            }}
            
            return html;
        }}
        
        
        function isContentNode(node) {{
            const tag = node.tag || '';
            const text = node.full_text || '';
            return tag === 'figure' || tag === 'table' || 
                   (tag === 'p' && text.length > 50) ||
                   (tag === 'ul' && node.metadata?.merged_count);
        }}
        
        function getNodeBadge(node) {{
            const badges = [];
            const metadata = node.metadata || {{}};
            
            if (metadata.heading_level) {{
                badges.push(`h${{metadata.heading_level}}`);
            }}
            
            if (metadata.merged_count) {{
                badges.push(`merged:${{metadata.merged_count}}`);
            }}
            
            if (node.children && node.children.length > 0) {{
                badges.push(`${{node.children.length}} children`);
            }}
            
            return badges.length > 0 ? `<span class="node-badge">${{badges.join(' | ')}}</span>` : '';
        }}
        
        function getContentIndicator(node) {{
            const tag = node.tag || '';
            if ((tag === 'figure' || tag === 'table') && (node.image_src || node.table_src)) {{
                return '<span class="content-indicator">🔍</span>';
            }}
            if ((tag === 'p' || tag === 'ul') && node.full_text && node.full_text.length > 50) {{
                return '<span class="content-indicator">🔍</span>';
            }}
            return '';
        }}
        
        function toggleNode(nodeId, event) {{
            if (event) {{
                event.stopPropagation();
            }}
            
            if (expandedNodes.has(nodeId)) {{
                expandedNodes.delete(nodeId);
            }} else {{
                expandedNodes.add(nodeId);
            }}
            
            renderTree();
        }}
        
        function selectNode(nodeId, event) {{
            event.stopPropagation();
            
            // Remove previous selection
            document.querySelectorAll('.tree-node.selected').forEach(node => {{
                node.classList.remove('selected');
            }});
            
            // Add new selection
            const nodeElement = document.querySelector(`[data-id="${{nodeId}}"]`);
            if (nodeElement) {{
                nodeElement.classList.add('selected');
                selectedNode = findNodeById(treeData, nodeId);
                
                // Show content if it's a content node
                if (selectedNode && isContentNode(selectedNode)) {{
                    showContent(selectedNode);
                }}
            }}
        }}
        
        function showContent(node) {{
            const modal = document.getElementById('contentModal');
            const content = document.getElementById('modalContent');
            const metadata = node.metadata || {{}};
            
            let contentHtml = `
                <h2>${{node.icon}} ${{node.tag}} Content Preview</h2>
                <p><strong>Node ID:</strong> ${{node.id || 'N/A'}}</p>
                <p><strong>Page:</strong> ${{metadata.page_number || 0}}</p>
            `;
            
            // Image content
            if (node.tag === 'figure' && node.image_src) {{
                contentHtml += `
                    <h3>🖼️ Image Content</h3>
                    <img src="${{node.image_src}}" alt="Image" class="preview-image" 
                         onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                    <p style="display: none; color: #666; font-style: italic;">Image not found: ${{node.image_src}}</p>
                `;
            }}
            
            // Table content
            if (node.tag === 'table' && node.table_src) {{
                contentHtml += `
                    <h3>📊 Table Content</h3>
                    <img src="${{node.table_src}}" alt="Table" class="preview-image" 
                         onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                    <p style="display: none; color: #666; font-style: italic;">Table image not found: ${{node.table_src}}</p>
                `;
            }}
            
            // Text content
            if (node.full_text && node.full_text.length > 50) {{
                contentHtml += `
                    <h3>📝 Text Content</h3>
                    <div class="text-content">${{node.full_text}}</div>
                `;
            }}
            
            // Merge information
            if (metadata.merged_count) {{
                contentHtml += `
                    <p style="margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 5px;">
                        <strong>ℹ️ Merge Info:</strong> This node contains ${{metadata.merged_count}} original elements
                    </p>
                `;
            }}
            
            content.innerHTML = contentHtml;
            modal.style.display = 'block';
        }}
        
        function closeModal() {{
            document.getElementById('contentModal').style.display = 'none';
        }}
        
        function findNodeById(node, id) {{
            if (!node) return null;
            if (node.id === id) return node;
            
            if (node.children) {{
                for (const child of node.children) {{
                    const result = findNodeById(child, id);
                    if (result) return result;
                }}
            }}
            
            return null;
        }}
        
        function expandToDepth(maxDepth) {{
            expandedNodes.clear();
            addNodesToDepth(treeData, 0, maxDepth);
            renderTree();
            
            // Update button state
            document.querySelectorAll('.controls button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            // Find and activate the correct button
            const buttons = document.querySelectorAll('.controls button');
            buttons.forEach(btn => {{
                if ((btn.textContent === 'Level 1' && maxDepth === 1) ||
                    (btn.textContent === 'Level 2' && maxDepth === 2) ||
                    (btn.textContent === 'Level 3' && maxDepth === 3)) {{
                    btn.classList.add('active');
                }}
            }});
        }}
        
        function addNodesToDepth(node, currentDepth, maxDepth) {{
            if (!node) return;
            
            if (currentDepth < maxDepth && node.children && node.children.length > 0) {{
                expandedNodes.add(node.id);
                for (const child of node.children) {{
                    addNodesToDepth(child, currentDepth + 1, maxDepth);
                }}
            }}
        }}
        
        function expandAll() {{
            expandedNodes.clear();
            addAllNodes(treeData);
            renderTree();
            
            document.querySelectorAll('.controls button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            // Find and activate the All button
            const buttons = document.querySelectorAll('.controls button');
            buttons.forEach(btn => {{
                if (btn.textContent === 'All') {{
                    btn.classList.add('active');
                }}
            }});
        }}
        
        function addAllNodes(node) {{
            if (!node) return;
            
            expandedNodes.add(node.id);
            if (node.children) {{
                for (const child of node.children) {{
                    addAllNodes(child);
                }}
            }}
        }}
        
        function collapseAll() {{
            expandedNodes.clear();
            renderTree();
            
            document.querySelectorAll('.controls button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            // Find and activate the Collapse button
            const buttons = document.querySelectorAll('.controls button');
            buttons.forEach(btn => {{
                if (btn.textContent === 'Collapse') {{
                    btn.classList.add('active');
                }}
            }});
        }}
        
        
        // Close modal when clicking outside
        window.onclick = function(event) {{
            const modal = document.getElementById('contentModal');
            if (event.target === modal) {{
                closeModal();
            }}
        }}
    </script>
</body>
</html>"""
        
        return html

def main():
    parser = argparse.ArgumentParser(description='DOM树结构交互式可视化工具')
    parser.add_argument('json_file', help='DOM树JSON文件路径')
    parser.add_argument('--output', '-o', default='interactive_tree.html', help='输出HTML文件名')
    parser.add_argument('--open', action='store_true', help='生成后自动打开浏览器')
    
    args = parser.parse_args()
    
    visualizer = DOMInteractiveTree(args.json_file)
    output_file = visualizer.generate_html(args.output)
    
    if args.open:
        file_path = os.path.abspath(output_file)
        webbrowser.open(f'file://{file_path}')
        print(f"🌐 已在浏览器中打开: {output_file}")

if __name__ == "__main__":
    main()