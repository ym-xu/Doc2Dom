#!/usr/bin/env python3
"""
简化检索评估系统 - 评估前10个节点中是否包含证据页面
使用CLIP模型，支持GPU加速，测试所有文档所有问题，保存结果
"""

import json
import os
import sys
import numpy as np
import time
from typing import Dict, List, Any
from collections import defaultdict
import ast
import traceback

# 添加项目路径
sys.path.append('.')

class SimpleRetrievalEvaluator:
    """简化检索评估器"""
    
    def __init__(self, samples_path: str = "samples.json", 
                 data_dir: str = "/data/users/yiming/dox2dom/dom/MMLongBench-Doc_qwen2vl-25-512"):
        self.samples_path = samples_path
        self.data_dir = data_dir
        self.samples = []
        self.embedder = None
        
        # 加载样本
        self._load_samples()
        
        # 结果存储
        self.results = {
            'overall_stats': {},
            'per_document_stats': {},
            'per_question_results': [],
            'evaluation_time': 0
        }
    
    def _load_samples(self):
        """加载样本数据"""
        print(f"📂 加载样本数据: {self.samples_path}")
        with open(self.samples_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        print(f"✅ 加载完成: {len(self.samples)} 个样本")
    
    def initialize_embedder(self):
        """初始化CLIP嵌入器"""
        print("🚀 初始化CLIP嵌入器...")
        from subtree_embedding import SubtreeEmbedder
        
        # 检查GPU可用性
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   使用设备: {device}")
        
        # 使用CLIP模型
        self.embedder = SubtreeEmbedder(
            model_name="openai/clip-vit-large-patch14",  # 使用CLIP
            aggregator="attention",
            embedding_dim=768,  # CLIP的维度
            data_dir=self.data_dir,
            device=device  # 传递设备信息
        )
        print(f"✅ 嵌入器初始化完成: {self.embedder.model_name}")
    
    def _load_document(self, doc_id: str) -> Dict:
        """加载文档DOM JSON"""
        doc_name = doc_id.replace('.pdf', '.json')
        doc_path = os.path.join(self.data_dir, doc_name)
        
        if not os.path.exists(doc_path):
            return None
        
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载文档失败 {doc_name}: {e}")
            return None
    
    def _collect_document_nodes(self, root_node) -> List[Dict]:
        """收集文档中的所有有效节点"""
        nodes = []
        
        def dfs_collect(node):
            if isinstance(node, dict):
                metadata = node.get('metadata', {})
                if (metadata.get('global_id') and 
                    metadata.get('page_number') is not None):
                    nodes.append(node)
                
                # 递归处理子节点
                for child in node.get('children', []):
                    dfs_collect(child)
        
        dfs_collect(root_node)
        return nodes
    
    def _embed_nodes_batch(self, nodes: List[Dict], batch_size: int = 50) -> Dict[str, Dict]:
        """批量编码节点"""
        node_embeddings = {}
        total_nodes = len(nodes)
        
        for i in range(0, total_nodes, batch_size):
            batch = nodes[i:i+batch_size]
            print(f"   编码节点批次: {i+1}-{min(i+batch_size, total_nodes)}/{total_nodes}")
            
            for node in batch:
                try:
                    metadata = node.get('metadata', {})
                    node_id = metadata.get('global_id')
                    page_number = metadata.get('page_number', 0)
                    element_type = metadata.get('element_type', 'unknown')
                    
                    # 编码节点
                    result = self.embedder.encode_subtree(node, max_depth=2)
                    
                    # 提取完整子树内容（不截断）
                    full_text_content = self._collect_subtree_text(node)
                    
                    # 为了向后兼容，保留100字符预览
                    content_preview = full_text_content[:100] if full_text_content else f"[{element_type}]"
                    
                    node_embeddings[node_id] = {
                        'embedding': result['embedding'],
                        'page_number': page_number,
                        'element_type': element_type,
                        'content_preview': content_preview,
                        'full_text_content': full_text_content,  # 完整子树文本内容
                        'original_node': node,  # 新增：保存原始节点完整信息
                        'metadata': metadata
                    }
                    
                except Exception as e:
                    # 静默失败，继续处理下一个节点
                    continue
        
        return node_embeddings
    
    def _retrieve_top_k(self, query: str, node_embeddings: Dict[str, Dict], k: int = 10) -> List[Dict]:
        """检索top-k相关节点"""
        if not node_embeddings:
            return []
        
        # 编码查询
        query_embedding = self.embedder.text_encoder.encode(query)
        
        # 计算相似度
        similarities = []
        for node_id, node_data in node_embeddings.items():
            try:
                node_embedding = node_data['embedding']
                similarity = np.dot(query_embedding, node_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding)
                )
                similarities.append({
                    'node_id': node_id,
                    'score': float(similarity),
                    'page_number': node_data['page_number'],
                    'element_type': node_data['element_type'],
                    'content_preview': node_data['content_preview']
                })
            except Exception:
                continue
        
        # 排序并返回top-k
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:k]
    
    def _extract_nodes_content(self, top_k_results: List[Dict], node_embeddings: Dict[str, Dict]) -> List[Dict]:
        """提取检索到的nodes的原始内容，供agents系统使用"""
        nodes_content = []
        
        for result in top_k_results:
            node_id = result['node_id']
            if node_id not in node_embeddings:
                continue
                
            node_data = node_embeddings[node_id]
            metadata = node_data.get('metadata', {})
            original_node = node_data.get('original_node', {})
            
            # 构建agents可用的节点内容
            node_content = {
                'node_id': node_id,
                'rank': len(nodes_content) + 1,
                'similarity_score': result['score'],
                'page_number': result['page_number'],
                'element_type': result['element_type'],
                
                # 文本内容
                'text_content': node_data.get('full_text_content', ''),
                
                # 完整子树结构信息
                'subtree_structure': self._extract_subtree_structure(original_node),
                
                # 位置信息
                'bbox': metadata.get('bbox'),
                'dom_path': self._construct_dom_path(metadata),
                
                # 结构信息
                'heading_level': metadata.get('heading_level'),
                'is_chapter_title': metadata.get('is_chapter_title', False),
                'parent_chapter': metadata.get('parent_chapter'),
                'depth': metadata.get('depth', 0),
                
                # 如果是图片，提取图片相关内容（包括链接）
                'image_info': self._extract_enhanced_image_content(original_node, metadata) if result['element_type'] == 'figure' else None,
                
                # 如果是表格，提取表格内容（包括完整表格数据）
                'table_info': self._extract_enhanced_table_content(original_node, metadata) if result['element_type'] == 'table' else None,
            }
            
            nodes_content.append(node_content)
        
        return nodes_content
    
    def _collect_subtree_text(self, node: Dict) -> str:
        """递归收集子树的所有文本内容"""
        if not isinstance(node, dict):
            return ""
        
        text_parts = []
        
        # 添加当前节点的文本
        node_text = node.get('text', '').strip()
        if node_text:
            text_parts.append(node_text)
        
        # 如果是图片，添加AI描述
        metadata = node.get('metadata', {})
        if metadata.get('element_type') == 'figure':
            ai_desc = metadata.get('ai_description', '')
            if ai_desc and ai_desc not in ['Skipped (disabled or too small)', 'Failed']:
                text_parts.append(f"[图片描述: {ai_desc}]")
        
        # 递归处理子节点
        children = node.get('children', [])
        for child in children:
            child_text = self._collect_subtree_text(child)
            if child_text:
                text_parts.append(child_text)
        
        return " ".join(text_parts)
    
    def _construct_dom_path(self, metadata: Dict) -> str:
        """构建DOM路径信息"""
        page = metadata.get('page_number', '?')
        if page != '?' and page is not None:
            page += 1  # 转换为1-based索引
        
        element_type = metadata.get('element_type', 'unknown')
        node_id = metadata.get('global_id', 'unknown')
        
        return f"Page {page} > {element_type}[{node_id}]"
    
    def _extract_image_content(self, metadata: Dict) -> Dict:
        """提取图片相关内容"""
        return {
            'ai_description': metadata.get('ai_description', ''),
            'description_method': metadata.get('description_method', ''),
            'image_extracted': metadata.get('image_extracted', False),
        }
    
    def _extract_table_content(self, metadata: Dict) -> Dict:
        """提取表格相关内容"""
        return {
            'table_image_extracted': metadata.get('table_image_extracted', False),
            'row_count': metadata.get('row_count'),
            'col_count': metadata.get('col_count'),
        }
    
    def _extract_subtree_structure(self, node: Dict) -> Dict:
        """提取子树结构信息"""
        if not isinstance(node, dict):
            return {}
        
        def extract_node_info(n):
            if not isinstance(n, dict):
                return None
            
            info = {
                'text': n.get('text', '').strip(),
                'metadata': n.get('metadata', {}),
                'children': []
            }
            
            # 递归处理子节点
            children = n.get('children', [])
            for child in children:
                child_info = extract_node_info(child)
                if child_info:
                    info['children'].append(child_info)
            
            return info
        
        return extract_node_info(node)
    
    def _extract_enhanced_image_content(self, node: Dict, metadata: Dict) -> Dict:
        """提取增强的图片内容信息"""
        image_info = {
            'ai_description': metadata.get('ai_description', ''),
            'description_method': metadata.get('description_method', ''),
            'image_extracted': metadata.get('image_extracted', False),
            'image_links': [],
            'image_path': '',
            'image_src': '',
        }
        
        # 提取图片src路径
        image_src = node.get('src', '')
        if image_src:
            image_info['image_src'] = image_src
            image_info['image_path'] = image_src  # 向后兼容
        
        # 从子树中提取图片链接和src信息
        def find_image_info(n):
            links = []
            src_paths = []
            
            if isinstance(n, dict):
                # 检查img标签的src
                if n.get('tag') == 'img' and n.get('src'):
                    src_paths.append(n['src'])
                
                # 检查节点自身的src
                if n.get('src'):
                    src_paths.append(n['src'])
                
                # 检查是否有链接信息
                if n.get('metadata', {}).get('href'):
                    links.append({
                        'href': n['metadata']['href'],
                        'text': n.get('text', '').strip()
                    })
                
                # 递归子节点
                for child in n.get('children', []):
                    child_links, child_srcs = find_image_info(child)
                    links.extend(child_links)
                    src_paths.extend(child_srcs)
            
            return links, src_paths
        
        links, src_paths = find_image_info(node)
        image_info['image_links'] = links
        
        # 如果没有直接的src，尝试从子节点获取
        if not image_info['image_src'] and src_paths:
            image_info['image_src'] = src_paths[0]  # 取第一个
            image_info['image_path'] = src_paths[0]
            
        # 保存所有找到的src路径
        image_info['all_image_sources'] = src_paths
        
        return image_info
    
    def _extract_enhanced_table_content(self, node: Dict, metadata: Dict) -> Dict:
        """提取增强的表格内容信息"""
        table_info = {
            'table_image_extracted': metadata.get('table_image_extracted', False),
            'row_count': metadata.get('row_count'),
            'col_count': metadata.get('col_count'),
            'table_data': [],
            'table_text': '',
            'table_src': '',
            'table_links': [],
        }
        
        # 提取表格src路径
        table_src = node.get('src', '')
        if table_src:
            table_info['table_src'] = table_src
        
        # 从子树中提取表格数据和链接
        def extract_table_info(n):
            if not isinstance(n, dict):
                return [], [], []
            
            rows = []
            table_text_parts = []
            src_paths = []
            
            # 检查src信息
            if n.get('src'):
                src_paths.append(n['src'])
            
            # 如果是表格行
            if n.get('metadata', {}).get('element_type') == 'table-row':
                row_cells = []
                for child in n.get('children', []):
                    if child.get('metadata', {}).get('element_type') == 'table-cell':
                        cell_text = self._collect_subtree_text(child)
                        row_cells.append(cell_text)
                        table_text_parts.append(cell_text)
                if row_cells:
                    rows.append(row_cells)
            
            # 递归处理子节点
            for child in n.get('children', []):
                child_rows, child_text, child_srcs = extract_table_info(child)
                rows.extend(child_rows)
                table_text_parts.extend(child_text)
                src_paths.extend(child_srcs)
            
            return rows, table_text_parts, src_paths
        
        table_data, text_parts, src_paths = extract_table_info(node)
        table_info['table_data'] = table_data
        table_info['table_text'] = ' | '.join(text_parts)
        
        # 如果没有直接的src，尝试从子节点获取
        if not table_info['table_src'] and src_paths:
            table_info['table_src'] = src_paths[0]
            
        # 保存所有找到的src路径
        table_info['all_table_sources'] = src_paths
        
        return table_info
    
    def _save_retrieved_nodes_for_agents(self, doc_id: str, question: str, nodes_content: List[Dict]):
        """保存检索到的nodes内容供agents系统使用"""
        # 创建保存目录
        agents_dir = "/data/users/yiming/dox2dom/retrieved_nodes/" + self.data_dir.split("/")[-1]
        os.makedirs(agents_dir, exist_ok=True)
        
        # 生成文件名（基于文档ID和问题hash）
        import hashlib
        question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        filename = f"{doc_id.replace('.pdf', '')}_{question_hash}.json"
        filepath = os.path.join(agents_dir, filename)
        
        # 构建保存的数据结构
        agents_data = {
            'doc_id': doc_id,
            'question': question,
            'retrieval_timestamp': time.time(),
            'total_nodes': len(nodes_content),
            'nodes_content': nodes_content
        }
        
        # 保存到JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(agents_data, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 保存检索内容供agents使用: {filepath}")
        return filepath
    
    def _get_retrieval_metric(self, gt_pages: List[int], pred_pages: List[int]) -> float:
        """计算检索指标 - MMLongBench风格的检索精度"""
        if not gt_pages:
            return 0.0
        
        # 对于每个ground truth页面，检查是否在预测页面中
        retrieval_precision_scores = []
        for gt_page in gt_pages:
            if gt_page in pred_pages:
                retrieval_precision_scores.append(1.0)
            else:
                retrieval_precision_scores.append(0.0)
        
        # 返回平均得分
        return sum(retrieval_precision_scores) / len(retrieval_precision_scores) if retrieval_precision_scores else 0.0
    
    def _get_similarity_score(self, chunk: str, answer: str) -> float:
        """计算chunk和答案之间的相似度分数"""
        chunk_lower = chunk.lower()
        answer_lower = answer.lower()
        
        # 精确匹配
        if answer_lower in chunk_lower:
            return 1.0
        
        # 词汇重叠分数
        chunk_words = set(chunk_lower.split())
        answer_words = set(answer_lower.split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(chunk_words & answer_words)
        return overlap / len(answer_words)
    
    def _eval_retrieval(self, gt_answers: List[str], retrieved_chunks: List[str]) -> Dict[str, float]:
        """使用chunk分数评估检索结果"""
        if not retrieved_chunks:
            return {"chunk_score": 0.0}
        
        scores = []
        for ans in gt_answers:
            ans_scores = [self._get_similarity_score(chunk, ans) for chunk in retrieved_chunks]
            best_score = max(ans_scores + [0])
            scores.append(np.log(best_score + 1) / np.log(2))
        
        return {
            "chunk_score": max(scores) if scores else 0.0
        }
    
    def _calculate_chunk_score(self, question_answer: str, top_10_results: List[Dict], node_embeddings: Dict[str, Dict]) -> float:
        """计算chunk分数 - 基于文本匹配的MMLongBench风格评估"""
        if not top_10_results or not question_answer or question_answer == "Not answerable":
            return 0.0
        
        # 提取检索到的chunks的文本内容
        retrieved_chunks = []
        for result in top_10_results:
            node_id = result['node_id']
            if node_id in node_embeddings:
                node_data = node_embeddings[node_id]
                chunk_text = node_data.get('full_text_content', '')
                if chunk_text.strip():
                    retrieved_chunks.append(chunk_text)
        
        # 使用ground truth答案评估
        gt_answers = [question_answer]
        chunk_metrics = self._eval_retrieval(gt_answers, retrieved_chunks)
        
        return chunk_metrics["chunk_score"]
    
    def _evaluate_question(self, sample: Dict, node_embeddings: Dict[str, Dict]) -> Dict[str, Any]:
        """评估单个问题"""
        # 解析证据页面
        try:
            evidence_pages = ast.literal_eval(sample['evidence_pages'])
            if not isinstance(evidence_pages, list):
                evidence_pages = [evidence_pages]
        except:
            evidence_pages = []
        
        # 解析证据来源
        try:
            evidence_sources = ast.literal_eval(sample.get('evidence_sources', '[]'))
            if not isinstance(evidence_sources, list):
                evidence_sources = [evidence_sources]
        except:
            evidence_sources = []
        
        # 执行检索（即使没有证据页面，也要检索供问答系统使用）
        query = sample['question']
        top_10_results = self._retrieve_top_k(query, node_embeddings, k=10)
        
        # 检查前10个节点中是否包含所有证据页面
        retrieved_pages = set()
        retrieved_sources = set()
        for result in top_10_results:
            page_num = result['page_number']
            # 添加0-based和1-based页面（兼容性检查）
            retrieved_pages.add(page_num + 1)
            
            # 根据element_type分类source
            element_type = result['element_type']
            if element_type == 'figure':
                retrieved_sources.add('Figure')
            elif element_type == 'table':
                retrieved_sources.add('Table')
            elif element_type in ['paragraph', 'text_block']:
                retrieved_sources.add('Pure-text (Plain-text)')
            elif element_type in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'heading']:
                retrieved_sources.add('Generalized-text (Layout)')
            else:
                retrieved_sources.add('Pure-text (Plain-text)')
        
        # 计算页面覆盖情况
        evidence_pages_set = set(evidence_pages)
        covered_pages = evidence_pages_set.intersection(retrieved_pages)
        
        # 计算页面级别的精确度、召回率和F1分数
        if evidence_pages_set and retrieved_pages:
            page_precision = len(covered_pages) / len(retrieved_pages)
            page_recall = len(covered_pages) / len(evidence_pages_set)
            page_f1 = 2 * page_precision * page_recall / (page_precision + page_recall) if (page_precision + page_recall) > 0 else 0.0
        else:
            page_precision = page_recall = page_f1 = 0.0
        
        # 计算源类型覆盖情况
        evidence_sources_set = set(evidence_sources)
        covered_sources = evidence_sources_set.intersection(retrieved_sources)
        
        # 如果没有证据页面，标记为不参与统计，但仍然提供检索结果
        has_evidence = len(evidence_pages_set) > 0
        coverage_ratio = len(covered_pages) / len(evidence_pages_set) if has_evidence else 0.0
        all_pages_covered = coverage_ratio == 1.0 if has_evidence else False
        
        # 计算源类型指标
        if evidence_sources_set:
            source_precision = len(covered_sources) / len(retrieved_sources) if retrieved_sources else 0
            source_recall = len(covered_sources) / len(evidence_sources_set)
            source_f1 = 2 * source_precision * source_recall / (source_precision + source_recall) if (source_precision + source_recall) > 0 else 0
        else:
            source_precision = source_recall = source_f1 = 0
        
        # 计算Hit@1, Hit@3, Hit@5 (只有有证据页面时才计算)
        hit_at_1 = False
        hit_at_3 = False  
        hit_at_5 = False
        
        if has_evidence and top_10_results:
            # 检查前1个
            page_1 = top_10_results[0]['page_number']
            hit_at_1 = page_1 in evidence_pages or (page_1 + 1) in evidence_pages
            
            # 检查前3个
            pages_3 = [r['page_number'] for r in top_10_results[:3]]
            hit_at_3 = any(p in evidence_pages or (p + 1) in evidence_pages for p in pages_3)
            
            # 检查前5个
            pages_5 = [r['page_number'] for r in top_10_results[:5]]
            hit_at_5 = any(p in evidence_pages or (p + 1) in evidence_pages for p in pages_5)
        
        # 计算新增指标
        # 1. Page hit rate: 是否有任何证据页面被检索到
        page_hit_rate = 1.0 if (evidence_pages_set & retrieved_pages) else 0.0 if has_evidence else 0.0
        
        # 2. Retrieval precision (page-based)
        gt_pages = list(evidence_pages_set)
        pred_pages = list(retrieved_pages)
        retrieval_precision = self._get_retrieval_metric(gt_pages, pred_pages) if gt_pages and pred_pages else 0
        
        # 3. Chunk score
        chunk_score = self._calculate_chunk_score(sample['answer'], top_10_results, node_embeddings)
        
        # 4. Top score (最高相似度分数)
        top_score = top_10_results[0]['score'] if top_10_results else 0
        
        # 提取并保存检索到的nodes内容供agents使用（包含完整子树内容）
        nodes_content = self._extract_nodes_content(top_10_results, node_embeddings)
        agents_file = self._save_retrieved_nodes_for_agents(sample['doc_id'], sample['question'], nodes_content)
        
        return {
            'doc_id': sample['doc_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'evidence_pages': evidence_pages,
            'evidence_sources': evidence_sources,
            'retrieved_pages': list(retrieved_pages),
            'retrieved_sources': list(retrieved_sources),
            'coverage_ratio': coverage_ratio,
            'all_pages_covered': all_pages_covered,
            'hit_at_1': hit_at_1,
            'hit_at_3': hit_at_3,
            'hit_at_5': hit_at_5,
            # 页面级别指标
            'page_precision': page_precision,
            'page_recall': page_recall,
            'page_f1': page_f1,
            # 源类型指标
            'source_precision': source_precision,
            'source_recall': source_recall,
            'source_f1': source_f1,
            # 其他指标
            'page_hit_rate': page_hit_rate,
            'retrieval_precision': retrieval_precision,
            'chunk_score': chunk_score,
            'top_score': top_score,
            'top_10_results': top_10_results,
            'agents_file': agents_file,  # 新增：保存的agents文件路径
            'has_evidence': has_evidence  # 新增：是否有证据页面（用于统计筛选）
        }
    
    def run_full_evaluation(self):
        """运行完整评估：所有文档，所有问题"""
        if not self.embedder:
            print("❌ 请先初始化嵌入器")
            return
        
        start_time = time.time()
        print(f"\n🎯 开始完整检索评估")
        print(f"总样本数: {len(self.samples)}")
        
        # 按文档分组样本
        samples_by_doc = defaultdict(list)
        for sample in self.samples:
            samples_by_doc[sample['doc_id']].append(sample)
        
        print(f"文档数: {len(samples_by_doc)}")
        
        # 总体统计
        total_questions = 0
        total_coverage = 0
        total_all_covered = 0
        total_hit_1 = 0
        total_hit_3 = 0
        total_hit_5 = 0
        total_skipped = 0  # 跳过的无证据页面样本数
        
        # 新增指标的总计
        total_page_precision = 0
        total_page_recall = 0
        total_page_f1 = 0
        total_source_precision = 0
        total_source_recall = 0
        total_source_f1 = 0
        total_page_hit_rate = 0
        total_retrieval_precision = 0
        total_chunk_score = 0
        total_top_score = 0
        
        # 逐文档处理
        for doc_idx, (doc_id, doc_samples) in enumerate(samples_by_doc.items()):
            print(f"\n📖 [{doc_idx+1}/{len(samples_by_doc)}] 处理文档: {doc_id}")
            print(f"   问题数: {len(doc_samples)}")
            
            # 加载文档
            document = self._load_document(doc_id)
            if not document:
                print(f"   ❌ 跳过文档（加载失败）")
                continue
            
            # 收集节点
            nodes = self._collect_document_nodes(document)
            if not nodes:
                print(f"   ❌ 跳过文档（无有效节点）")
                continue
            
            print(f"   节点数: {len(nodes)}")
            
            # 编码节点
            try:
                node_embeddings = self._embed_nodes_batch(nodes)
                if not node_embeddings:
                    print(f"   ❌ 跳过文档（编码失败）")
                    continue
                print(f"   ✅ 成功编码: {len(node_embeddings)} 个节点")
            except Exception as e:
                print(f"   ❌ 跳过文档（编码异常）: {e}")
                continue
            
            # 评估该文档的所有问题
            doc_results = []
            doc_coverage = 0
            doc_all_covered = 0
            doc_hit_1 = 0
            doc_hit_3 = 0
            doc_hit_5 = 0
            
            # 文档级别的新增指标
            doc_page_precision = 0
            doc_page_recall = 0
            doc_page_f1 = 0
            doc_source_precision = 0
            doc_source_recall = 0
            doc_source_f1 = 0
            doc_page_hit_rate = 0
            doc_retrieval_precision = 0
            doc_chunk_score = 0
            doc_top_score = 0
            
            for sample_idx, sample in enumerate(doc_samples):
                if sample_idx % 10 == 0:
                    print(f"   评估进度: {sample_idx+1}/{len(doc_samples)}")
                
                try:
                    result = self._evaluate_question(sample, node_embeddings)
                    
                    if result is None:
                        continue
                    
                    doc_results.append(result)
                    self.results['per_question_results'].append(result)
                    
                    # 只有有证据页面的样本才参与统计计算
                    if result['has_evidence']:
                        doc_coverage += result['coverage_ratio']
                        if result['all_pages_covered']:
                            doc_all_covered += 1
                        if result['hit_at_1']:
                            doc_hit_1 += 1
                        if result['hit_at_3']:
                            doc_hit_3 += 1
                        if result['hit_at_5']:
                            doc_hit_5 += 1
                            
                        # 累计新增指标
                        doc_page_precision += result['page_precision']
                        doc_page_recall += result['page_recall']
                        doc_page_f1 += result['page_f1']
                        doc_source_precision += result['source_precision']
                        doc_source_recall += result['source_recall']
                        doc_source_f1 += result['source_f1']
                        doc_page_hit_rate += result['page_hit_rate']
                        doc_retrieval_precision += result['retrieval_precision']
                        doc_chunk_score += result['chunk_score']
                        doc_top_score += result['top_score']
                    else:
                        total_skipped += 1
                        print(f"   ⚠️  跳过统计（无证据页面）: {sample['question'][:50]}...")
                    
                except Exception as e:
                    print(f"   ❌ 问题评估失败: {e}")
                    continue
            
            # 文档级别统计
            if doc_results:
                # 计算有证据页面的样本数量
                doc_valid_count = sum(1 for r in doc_results if r['has_evidence'])
                
                doc_stats = {
                    'questions_count': len(doc_results),
                    'valid_questions_count': doc_valid_count,
                    'skipped_questions_count': len(doc_results) - doc_valid_count,
                    'avg_coverage': doc_coverage / doc_valid_count if doc_valid_count > 0 else 0.0,
                    'all_pages_covered_rate': doc_all_covered / doc_valid_count if doc_valid_count > 0 else 0.0,
                    'hit_at_1_rate': doc_hit_1 / doc_valid_count if doc_valid_count > 0 else 0.0,
                    'hit_at_3_rate': doc_hit_3 / doc_valid_count if doc_valid_count > 0 else 0.0,
                    'hit_at_5_rate': doc_hit_5 / doc_valid_count if doc_valid_count > 0 else 0.0,
                    # 页面级别指标
                    'avg_page_precision': doc_page_precision / doc_valid_count if doc_valid_count > 0 else 0.0,
                    'avg_page_recall': doc_page_recall / doc_valid_count if doc_valid_count > 0 else 0.0,
                    'avg_page_f1': doc_page_f1 / doc_valid_count if doc_valid_count > 0 else 0.0,
                    # 源类型指标
                    'avg_source_precision': doc_source_precision / doc_valid_count if doc_valid_count > 0 else 0.0,
                    'avg_source_recall': doc_source_recall / doc_valid_count if doc_valid_count > 0 else 0.0,
                    'avg_source_f1': doc_source_f1 / doc_valid_count if doc_valid_count > 0 else 0.0,
                    # 其他指标
                    'avg_page_hit_rate': doc_page_hit_rate / doc_valid_count if doc_valid_count > 0 else 0.0,
                    'avg_retrieval_precision': doc_retrieval_precision / doc_valid_count if doc_valid_count > 0 else 0.0,
                    'avg_chunk_score': doc_chunk_score / doc_valid_count if doc_valid_count > 0 else 0.0,
                    'avg_top_score': doc_top_score / doc_valid_count if doc_valid_count > 0 else 0.0
                }
                self.results['per_document_stats'][doc_id] = doc_stats
                
                # 累计到总体统计（只计算有证据页面的样本）
                total_questions += doc_valid_count
                total_coverage += doc_coverage
                total_all_covered += doc_all_covered
                total_hit_1 += doc_hit_1
                total_hit_3 += doc_hit_3
                total_hit_5 += doc_hit_5
                
                # 累计新增指标
                total_page_precision += doc_page_precision
                total_page_recall += doc_page_recall
                total_page_f1 += doc_page_f1
                total_source_precision += doc_source_precision
                total_source_recall += doc_source_recall
                total_source_f1 += doc_source_f1
                total_page_hit_rate += doc_page_hit_rate
                total_retrieval_precision += doc_retrieval_precision
                total_chunk_score += doc_chunk_score
                total_top_score += doc_top_score
                
                print(f"   📊 文档统计: 覆盖率={doc_stats['avg_coverage']:.3f}, "
                      f"完全覆盖={doc_stats['all_pages_covered_rate']:.3f}, "
                      f"Hit@1={doc_stats['hit_at_1_rate']:.3f}")
        
        # 计算总体统计
        if total_questions > 0:
            self.results['overall_stats'] = {
                'total_questions': total_questions,
                'total_documents': len(self.results['per_document_stats']),
                'total_skipped': total_skipped,
                'total_samples': len(self.samples),
                'avg_coverage': total_coverage / total_questions,
                'all_pages_covered_rate': total_all_covered / total_questions,
                'hit_at_1_rate': total_hit_1 / total_questions,
                'hit_at_3_rate': total_hit_3 / total_questions,
                'hit_at_5_rate': total_hit_5 / total_questions,
                # 页面级别指标
                'avg_page_precision': total_page_precision / total_questions,
                'avg_page_recall': total_page_recall / total_questions,
                'avg_page_f1': total_page_f1 / total_questions,
                # 源类型指标
                'avg_source_precision': total_source_precision / total_questions,
                'avg_source_recall': total_source_recall / total_questions,
                'avg_source_f1': total_source_f1 / total_questions,
                # 其他指标
                'avg_page_hit_rate': total_page_hit_rate / total_questions,
                'avg_retrieval_precision': total_retrieval_precision / total_questions,
                'avg_chunk_score': total_chunk_score / total_questions,
                'avg_top_score': total_top_score / total_questions
            }
        
        # 记录评估时间
        self.results['evaluation_time'] = time.time() - start_time
        
        # 显示结果
        self._display_results()
        
        # 保存结果
        self._save_results()
        
        # 显示agents文件保存信息
        agents_files_count = sum(1 for r in self.results['per_question_results'] if r.get('agents_file'))
        valid_agents_count = sum(1 for r in self.results['per_question_results'] if r.get('agents_file') and r['has_evidence'])
        no_evidence_agents_count = sum(1 for r in self.results['per_question_results'] if r.get('agents_file') and not r['has_evidence'])
        
        print(f"\n💾 已为 {agents_files_count} 个问题保存检索内容到 ./retrieved_nodes_for_agents/ 目录")
        print(f"   - 有证据页面: {valid_agents_count} 个")
        print(f"   - 无证据页面: {no_evidence_agents_count} 个")
        print("   所有文件都可供agents系统使用（问答系统可据此判断是否有足够信息回答）")
    
    def _display_results(self):
        """显示评估结果"""
        stats = self.results['overall_stats']
        if not stats:
            print("❌ 无评估结果")
            return
        
        print("\n" + "="*80)
        print("🎯 完整检索评估结果")
        print("="*80)
        
        print(f"\n📊 总体统计:")
        print(f"   总样本数: {stats['total_samples']}")
        print(f"   有效问题数: {stats['total_questions']} (有证据页面)")
        print(f"   跳过样本数: {stats['total_skipped']} (无证据页面，但已检索)")
        print(f"   处理文档数: {stats['total_documents']}")
        print(f"   评估时间: {self.results['evaluation_time']:.1f}秒")
        
        print(f"\n📈 检索性能指标:")
        print(f"   平均页面覆盖率: {stats['avg_coverage']:.3f}")
        print(f"   完全覆盖率: {stats['all_pages_covered_rate']:.3f}")
        print(f"   Hit@1: {stats['hit_at_1_rate']:.3f}")
        print(f"   Hit@3: {stats['hit_at_3_rate']:.3f}")
        print(f"   Hit@5: {stats['hit_at_5_rate']:.3f}")
        
        print(f"\n📊 页面级别指标:")
        print(f"   页面精确度: {stats['avg_page_precision']:.3f}")
        print(f"   页面召回率: {stats['avg_page_recall']:.3f}")
        print(f"   页面F1分数: {stats['avg_page_f1']:.3f}")
        print(f"   页面命中率: {stats['avg_page_hit_rate']:.3f}")
        
        print(f"\n📊 源类型指标:")
        print(f"   源类型精度: {stats['avg_source_precision']:.3f}")
        print(f"   源类型召回: {stats['avg_source_recall']:.3f}")
        print(f"   源类型F1: {stats['avg_source_f1']:.3f}")
        
        print(f"\n📊 其他评估指标:")
        print(f"   检索精度: {stats['avg_retrieval_precision']:.3f}")
        print(f"   块得分: {stats['avg_chunk_score']:.3f}")
        print(f"   最高得分: {stats['avg_top_score']:.3f}")
        
        # 显示一些成功和失败案例（只考虑有证据页面的样本）
        valid_results = [r for r in self.results['per_question_results'] if r['has_evidence']]
        success_cases = [r for r in valid_results if r['all_pages_covered']]
        fail_cases = [r for r in valid_results if not r['all_pages_covered']]
        no_evidence_cases = [r for r in self.results['per_question_results'] if not r['has_evidence']]
        
        if success_cases:
            print(f"\n✅ 成功案例 ({len(success_cases)}个):")
            for case in success_cases[:2]:
                print(f"   文档: {case['doc_id']}")
                print(f"   问题: {case['question'][:60]}...")
                print(f"   证据页面: {case['evidence_pages']}")
                print(f"   检索页面: {sorted(list(set(case['retrieved_pages'])))[:10]}")
                print()
        
        if fail_cases:
            print(f"\n❌ 失败案例 ({len(fail_cases)}个):")
            for case in fail_cases[:2]:
                print(f"   文档: {case['doc_id']}")
                print(f"   问题: {case['question'][:60]}...")
                print(f"   证据页面: {case['evidence_pages']}")
                print(f"   检索页面: {sorted(list(set(case['retrieved_pages'])))[:10]}")
                print(f"   覆盖率: {case['coverage_ratio']:.3f}")
                print()
        
        if no_evidence_cases:
            print(f"\n⚠️  无证据页面案例 ({len(no_evidence_cases)}个，已检索但未参与统计):")
            for case in no_evidence_cases[:2]:
                print(f"   文档: {case['doc_id']}")
                print(f"   问题: {case['question'][:60]}...")
                print(f"   检索页面: {sorted(list(set(case['retrieved_pages'])))[:10]}")
                print(f"   agents文件: {case.get('agents_file', 'N/A')}")
                print()
    
    def _save_results(self):
        """保存评估结果"""
        output_file = "/data/users/yiming/dox2dom/retrieved_nodes/" + self.data_dir.split("/")[-1] + "/domretrieval_evaluation_results.json"
        
        # 为了JSON序列化，转换numpy数组等
        serializable_results = {
            'overall_stats': self.results['overall_stats'],
            'per_document_stats': self.results['per_document_stats'],
            'evaluation_time': self.results['evaluation_time'],
            'model_info': {
                'model_name': self.embedder.model_name if self.embedder else 'unknown',
                'embedding_dim': self.embedder.text_encoder.target_dim if self.embedder else 'unknown'
            },
            'evaluation_config': {
                'top_k': 10,
                'max_depth': 2,
                'data_dir': self.data_dir
            }
        }
        
        # 保存简化版的问题结果（去掉embedding等大对象）
        simplified_question_results = []
        for result in self.results['per_question_results']:
            simplified_result = {
                'doc_id': result['doc_id'],
                'question': result['question'][:200],  # 截断问题
                'evidence_pages': result['evidence_pages'],
                'coverage_ratio': result['coverage_ratio'],
                'all_pages_covered': result['all_pages_covered'],
                'hit_at_1': result['hit_at_1'],
                'hit_at_3': result['hit_at_3'],
                'hit_at_5': result['hit_at_5'],
                'top_3_results': [
                    {
                        'page_number': r['page_number'],
                        'score': r['score'],
                        'element_type': r['element_type']
                    }
                    for r in result['top_10_results'][:3]
                ]
            }
            simplified_question_results.append(simplified_result)
        
        serializable_results['per_question_results'] = simplified_question_results
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 结果已保存到: {output_file}")
        except Exception as e:
            print(f"\n❌ 保存结果失败: {e}")

def main():
    """主函数"""
    print("🚀 启动简化检索评估系统")
    
    # 创建评估器
    evaluator = SimpleRetrievalEvaluator()
    
    # 初始化嵌入器
    evaluator.initialize_embedder()
    
    # 运行完整评估
    evaluator.run_full_evaluation()
    
    print("\n🎉 评估完成!")

if __name__ == "__main__":
    main()