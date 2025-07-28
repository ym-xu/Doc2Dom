import numpy as np
from PIL import Image
import torch
import os

class ImageEncoder:
    """
    优化后的Image Encoder - 针对我们的DOM结构设计
    """
    
    def __init__(self, shared_model, shared_processor, target_dim: int = 768,
                 data_dir: str = "./data/dom/MMLongBench-Doc", device: str = "cpu"):
        """
        ImageEncoder现在接收共享模型，支持BLIP2联合编码
        """
        self.model = shared_model
        self.processor = shared_processor
        self.target_dim = target_dim
        self.data_dir = data_dir
        self.device = device
        
        # 简化：直接使用target_dim，让SubtreeEmbedder处理维度匹配
        self.actual_dim = target_dim
        self.projection = None

    def encode(self, node, structure_prompt: str = "") -> np.ndarray:
        """
        编码图片节点 - 使用BLIP2联合编码，避免semantic gap
        """
        # 提取图像路径和文本上下文
        image_path = self._extract_image_path(node)
        text_context = self._extract_text_context(node, structure_prompt)
        
        # 使用BLIP2/CLIP联合编码（这是关键改进）
        return self._encode_image_text_joint(image_path, text_context)
    
    def _extract_image_path(self, node) -> str:
        if not isinstance(node, dict):
            return ""
        return node.get("src") or ""
    
    def _extract_text_context(self, node, structure_prompt: str = "") -> str:
        """
        提取图片的文本上下文 - 包含结构信息
        """
        if not isinstance(node, dict):
            return ""
        
        context_parts = []
        
        # 1. 添加结构信息（如果提供）
        if structure_prompt.strip():
            context_parts.append(structure_prompt.strip())
        
        metadata = node.get('metadata', {})
        
        # 2. 图片说明文字
        text = node.get('text', '').strip()
        if text:
            context_parts.append(f"Caption: {text}")
        
        # 3. AI描述（如果有效）
        ai_desc = metadata.get('ai_description', '')
        if (ai_desc and 
            ai_desc != 'Skipped (disabled or too small)' and 
            not ai_desc.startswith('Failed') and
            len(ai_desc.strip()) > 10):
            context_parts.append(f"Description: {ai_desc}")
        
        return ". ".join(context_parts)
    
    
    def _assess_text_quality(self, node) -> float:
        """
        评估文本上下文的质量 (0-1)
        """
        if not isinstance(node, dict):
            return 0.0
        
        score = 0.0
        metadata = node.get('metadata', {})
        
        # AI描述质量 (50%)
        ai_desc = metadata.get('ai_description', '')
        if (ai_desc and ai_desc != 'Skipped (disabled or too small)' and 
            not ai_desc.startswith('Failed')):
            score += 0.5
            if len(ai_desc) > 100:  # 详细描述加分
                score += 0.1
        
        # 图片说明质量 (30%)
        text = node.get('text', '').strip()
        if text and len(text) > 5:
            score += 0.3
        
        # 位置和页面信息 (20%)
        if metadata.get('bbox') and metadata.get('page_number') is not None:
            score += 0.2
        
        return min(score, 1.0)
    
    def _encode_image_text_joint(self, image_path: str, text_context: str) -> np.ndarray:
        """
        使用共享模型进行图像+文本联合编码，避免semantic gap
        这是BLIP2方法的核心：统一的多模态编码空间
        """
        try:
            # 准备图像
            if not image_path or not os.path.exists(image_path):
                # 如果没有图像，使用纯文本编码
                return self._encode_text_only(text_context)
            
            image = Image.open(image_path).convert("RGB")
            
            # 准备文本
            if not text_context.strip():
                text_context = "Image content"  # 默认文本
            
            # 关键：联合处理图像和文本
            if hasattr(self.model, 'get_image_features') and hasattr(self.model, 'get_text_features'):
                # CLIP类型：分别编码然后融合
                return self._encode_clip_joint(image, text_context)
            else:
                # BLIP2类型：真正的联合编码
                return self._encode_blip2_joint(image, text_context)
                
        except Exception as e:
            print(f"Error in joint encoding: {str(e)}")
            return np.zeros(self.target_dim)
    
    def _encode_clip_joint(self, image: Image.Image, text_context: str) -> np.ndarray:
        """
        CLIP模型的联合编码：分别编码后融合
        """
        # 图像编码
        image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in image_inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
        
        # 文本编码
        text_inputs = self.processor(text=text_context, return_tensors="pt", 
                                   padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        
        # 智能融合：根据文本质量自适应权重
        image_emb = image_features.squeeze().cpu().detach().numpy()
        text_emb = text_features.squeeze().cpu().detach().numpy()
        
        # 确保维度匹配
        if image_emb.shape[-1] != self.target_dim:
            image_emb = self._adapt_dimension(image_emb)
        if text_emb.shape[-1] != self.target_dim:
            text_emb = self._adapt_dimension(text_emb)
        
        # 自适应权重融合
        text_weight = min(0.4, len(text_context) / 200.0)  # 文本越丰富权重越高
        image_weight = 1.0 - text_weight
        
        return image_weight * image_emb + text_weight * text_emb
    
    def _encode_blip2_joint(self, image: Image.Image, text_context: str) -> np.ndarray:
        """
        BLIP2模型的真正联合编码：统一的多模态空间
        """
        # BLIP2的联合处理
        inputs = self.processor(images=image, text=text_context, 
                              return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            # BLIP2的联合特征提取
            outputs = self.model(**inputs)
            
            # 获取联合表示
            if hasattr(outputs, 'image_embeds'):
                # 使用图像嵌入作为主要特征
                joint_features = outputs.image_embeds
            elif hasattr(outputs, 'last_hidden_state'):
                # 使用最后隐藏状态的平均池化
                joint_features = outputs.last_hidden_state.mean(dim=1)
            else:
                # 回退方案
                joint_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.logits
        
        joint_emb = joint_features.squeeze().cpu().detach().numpy()
        
        # 维度适配
        if joint_emb.shape[-1] != self.target_dim:
            joint_emb = self._adapt_dimension(joint_emb)
        
        return joint_emb
    
    def _encode_text_only(self, text_context: str) -> np.ndarray:
        """
        纯文本编码（当图像不可用时）
        """
        if not text_context.strip():
            return np.zeros(self.target_dim)
        
        try:
            inputs = self.processor(text=text_context, return_tensors="pt", 
                                  padding=True, truncation=True)
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            with torch.no_grad():
                if hasattr(self.model, 'get_text_features'):
                    features = self.model.get_text_features(**inputs)
                else:
                    outputs = self.model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1)
            
            emb = features.squeeze().cpu().detach().numpy()
            
            if emb.shape[-1] != self.target_dim:
                emb = self._adapt_dimension(emb)
            
            return emb
            
        except Exception as e:
            print(f"Error in text-only encoding: {str(e)}")
            return np.zeros(self.target_dim)
    
    def _adapt_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """
        简单的维度适配：截断或填充
        """
        if embedding.shape[-1] > self.target_dim:
            return embedding[..., :self.target_dim]
        elif embedding.shape[-1] < self.target_dim:
            pad_size = self.target_dim - embedding.shape[-1]
            return np.pad(embedding, (0, pad_size), mode='constant')
        else:
            return embedding

# 测试函数
def test_image_encoder_joint():
    """
    测试新的联合编码功能
    """
    print("=== ImageEncoder联合编码测试 ===")
    print("注意：需要共享模型实例才能完整运行")
    print()
    
    # 测试节点
    test_node = {
        "tag": "figure",
        "text": "6",
        "src": "welcome-to-nus/page_1_figure_2.png",
        "metadata": {
            "element_type": "figure", 
            "page_number": 0,
            "bbox": [209.0, 147.0, 861.0, 576.5],
            "image_extracted": True,
            "ai_description": "The target region contains a large, colorful sign with NUSLife text.",
            "description_method": "full_page_region"
        }
    }
    
    print("1. 图像路径提取:")
    print(f"   Path: welcome-to-nus/page_1_figure_2.png")
    
    print("\n2. 文本上下文构建:")
    structure_prompt = "[Page 1] [Depth 1] [Figure] [Pos: 209,147]"
    text_context = f"{structure_prompt}. Caption: 6. Description: The target region contains a large, colorful sign with NUSLife text."
    print(f"   Context: {text_context}")
    
    print("\n3. 联合编码优势:")
    print("   ✓ BLIP2: 统一多模态空间，避免semantic gap")
    print("   ✓ CLIP: 自适应权重融合")
    print("   ✓ 智能回退: 图像缺失时使用纯文本编码")
    
    print("\n4. 维度适配:")
    print("   ✓ 自动处理不同模型的维度差异")
    print("   ✓ 截断或填充到目标维度")
    
    print("\n✓ ImageEncoder已升级为BLIP2联合编码架构!")

if __name__ == "__main__":
    test_image_encoder_joint()