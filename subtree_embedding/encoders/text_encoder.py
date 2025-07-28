import numpy as np
from typing import Union, List
import torch
import hashlib

class TextEncoder:
    """
    简化的TextEncoder - 只支持CLIP模型的文本编码
    """

    def __init__(self, shared_model, shared_processor, target_dim: int = 768, device: str = "cpu"):
        """
        TextEncoder现在只接收已加载的共享模型，不再独立加载模型
        """
        self.model = shared_model
        self.processor = shared_processor
        self.target_dim = target_dim
        self.device = device
        
        # 简化：直接使用target_dim，让SubtreeEmbedder处理维度匹配
        self.actual_dim = target_dim
        self.projection = None  # 不再需要投影层

        # 文本缓存
        self.cache = {}
        self.cache_size = 1000

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        编码文本为向量
        """
        if not text or (isinstance(text, str) and not text.strip()):
            return np.zeros(self.target_dim)
        
        # 缓存检查
        if isinstance(text, str):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.cache:
                return self.cache[text_hash]
        
        # 使用共享模型进行编码（自动适配CLIP/BLIP2）
        embeddings = self._encode_with_shared_model(text)

        # 更新缓存
        if isinstance(text, str):
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[text_hash] = embeddings.copy()

        # 确保返回正确的维度
        if embeddings.ndim == 1:
            return embeddings
        else:
            return embeddings[0]
    
    def _encode_with_shared_model(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        使用共享模型编码文本（自动适配CLIP/BLIP2等）
        """
        if isinstance(text, str):
            text = [text]
        
        try:
            # 处理文本输入
            inputs = self.processor(text=text, return_tensors="pt", 
                                  padding=True, truncation=True)
            # 将输入移动到正确的设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # 根据模型类型选择合适的方法
                if hasattr(self.model, 'get_text_features'):
                    # CLIP类型模型
                    text_features = self.model.get_text_features(**inputs)
                elif hasattr(self.model, 'language_model'):
                    # BLIP2类型模型 - 使用language_model的encoder部分
                    text_inputs = {k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
                    # 使用T5的encoder部分
                    encoder_outputs = self.model.language_model.encoder(**text_inputs)
                    text_features = encoder_outputs.last_hidden_state.mean(dim=1)
                else:
                    # 通用回退方法
                    outputs = self.model(**inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        text_features = outputs.last_hidden_state.mean(dim=1)
                    else:
                        text_features = outputs.pooler_output
            
            # 转换为numpy
            embeddings = text_features.detach().cpu().numpy()
            
            # 确保维度正确
            if embeddings.shape[-1] != self.target_dim:
                # 简单的维度适配：截断或填充
                if embeddings.shape[-1] > self.target_dim:
                    embeddings = embeddings[..., :self.target_dim]
                else:
                    pad_size = self.target_dim - embeddings.shape[-1]
                    embeddings = np.pad(embeddings, ((0, 0), (0, pad_size)), mode='constant')
            
            if len(text) == 1:
                return embeddings[0]
            else:
                return embeddings
                
        except Exception as e:
            print(f"Error encoding text with shared model: {str(e)}")
            return np.zeros((len(text), self.target_dim)) if isinstance(text, list) else np.zeros(self.target_dim)