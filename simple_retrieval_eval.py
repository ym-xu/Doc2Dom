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
                 data_dir: str = "./data/dom/MMLongBench-Doc"):
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
                    
                    # 提取内容预览
                    text = node.get('text', '').strip()
                    ai_desc = metadata.get('ai_description', '')
                    content_preview = text[:100] if text else f"[{element_type}]"
                    
                    node_embeddings[node_id] = {
                        'embedding': result['embedding'],
                        'page_number': page_number,
                        'element_type': element_type,
                        'content_preview': content_preview,
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
    
    def _evaluate_question(self, sample: Dict, node_embeddings: Dict[str, Dict]) -> Dict[str, Any]:
        """评估单个问题"""
        # 解析证据页面
        try:
            evidence_pages = ast.literal_eval(sample['evidence_pages'])
            if not isinstance(evidence_pages, list):
                evidence_pages = [evidence_pages]
        except:
            evidence_pages = []
        
        # 执行检索
        query = sample['question']
        top_10_results = self._retrieve_top_k(query, node_embeddings, k=10)
        
        # 检查前10个节点中是否包含所有证据页面
        retrieved_pages = set()
        for result in top_10_results:
            page_num = result['page_number']
            # 添加0-based和1-based页面（兼容性检查）
            # retrieved_pages.add(page_num)
            retrieved_pages.add(page_num + 1)
        
        # 计算覆盖情况
        evidence_pages_set = set(evidence_pages)
        covered_pages = evidence_pages_set.intersection(retrieved_pages)
        coverage_ratio = len(covered_pages) / len(evidence_pages_set) if evidence_pages_set else 0.0
        all_pages_covered = coverage_ratio == 1.0
        
        # 计算Hit@1, Hit@3, Hit@5
        hit_at_1 = False
        hit_at_3 = False  
        hit_at_5 = False
        
        if top_10_results:
            # 检查前1个
            page_1 = top_10_results[0]['page_number']
            hit_at_1 = page_1 in evidence_pages or (page_1 + 1) in evidence_pages
            
            # 检查前3个
            pages_3 = [r['page_number'] for r in top_10_results[:3]]
            hit_at_3 = any(p in evidence_pages or (p + 1) in evidence_pages for p in pages_3)
            
            # 检查前5个
            pages_5 = [r['page_number'] for r in top_10_results[:5]]
            hit_at_5 = any(p in evidence_pages or (p + 1) in evidence_pages for p in pages_5)
        
        return {
            'doc_id': sample['doc_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'evidence_pages': evidence_pages,
            'evidence_sources': sample.get('evidence_sources', ''),
            'retrieved_pages': list(retrieved_pages),
            'coverage_ratio': coverage_ratio,
            'all_pages_covered': all_pages_covered,
            'hit_at_1': hit_at_1,
            'hit_at_3': hit_at_3,
            'hit_at_5': hit_at_5,
            'top_10_results': top_10_results
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
            
            for sample_idx, sample in enumerate(doc_samples):
                if sample_idx % 10 == 0:
                    print(f"   评估进度: {sample_idx+1}/{len(doc_samples)}")
                
                try:
                    result = self._evaluate_question(sample, node_embeddings)
                    doc_results.append(result)
                    self.results['per_question_results'].append(result)
                    
                    # 累计统计
                    doc_coverage += result['coverage_ratio']
                    if result['all_pages_covered']:
                        doc_all_covered += 1
                    if result['hit_at_1']:
                        doc_hit_1 += 1
                    if result['hit_at_3']:
                        doc_hit_3 += 1
                    if result['hit_at_5']:
                        doc_hit_5 += 1
                    
                except Exception as e:
                    print(f"   ❌ 问题评估失败: {e}")
                    continue
            
            # 文档级别统计
            if doc_results:
                doc_stats = {
                    'questions_count': len(doc_results),
                    'avg_coverage': doc_coverage / len(doc_results),
                    'all_pages_covered_rate': doc_all_covered / len(doc_results),
                    'hit_at_1_rate': doc_hit_1 / len(doc_results),
                    'hit_at_3_rate': doc_hit_3 / len(doc_results),
                    'hit_at_5_rate': doc_hit_5 / len(doc_results)
                }
                self.results['per_document_stats'][doc_id] = doc_stats
                
                # 累计到总体统计
                total_questions += len(doc_results)
                total_coverage += doc_coverage
                total_all_covered += doc_all_covered
                total_hit_1 += doc_hit_1
                total_hit_3 += doc_hit_3
                total_hit_5 += doc_hit_5
                
                print(f"   📊 文档统计: 覆盖率={doc_stats['avg_coverage']:.3f}, "
                      f"完全覆盖={doc_stats['all_pages_covered_rate']:.3f}, "
                      f"Hit@1={doc_stats['hit_at_1_rate']:.3f}")
        
        # 计算总体统计
        if total_questions > 0:
            self.results['overall_stats'] = {
                'total_questions': total_questions,
                'total_documents': len(self.results['per_document_stats']),
                'avg_coverage': total_coverage / total_questions,
                'all_pages_covered_rate': total_all_covered / total_questions,
                'hit_at_1_rate': total_hit_1 / total_questions,
                'hit_at_3_rate': total_hit_3 / total_questions,
                'hit_at_5_rate': total_hit_5 / total_questions
            }
        
        # 记录评估时间
        self.results['evaluation_time'] = time.time() - start_time
        
        # 显示结果
        self._display_results()
        
        # 保存结果
        self._save_results()
    
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
        print(f"   总问题数: {stats['total_questions']}")
        print(f"   处理文档数: {stats['total_documents']}")
        print(f"   评估时间: {self.results['evaluation_time']:.1f}秒")
        
        print(f"\n📈 检索性能指标:")
        print(f"   平均页面覆盖率: {stats['avg_coverage']:.3f}")
        print(f"   完全覆盖率: {stats['all_pages_covered_rate']:.3f}")
        print(f"   Hit@1: {stats['hit_at_1_rate']:.3f}")
        print(f"   Hit@3: {stats['hit_at_3_rate']:.3f}")
        print(f"   Hit@5: {stats['hit_at_5_rate']:.3f}")
        
        # 显示一些成功和失败案例
        success_cases = [r for r in self.results['per_question_results'] if r['all_pages_covered']]
        fail_cases = [r for r in self.results['per_question_results'] if not r['all_pages_covered']]
        
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
    
    def _save_results(self):
        """保存评估结果"""
        output_file = "retrieval_evaluation_results.json"
        
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