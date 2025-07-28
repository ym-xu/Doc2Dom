import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class Aggregator:
    """
    Aggregator - aggregate self embedding and child embeddings
    """

    def __init__(self, strategy: str = 'attention', embedding_dim: int = 768):
        self.strategy = strategy
        self.embedding_dim = embedding_dim

        if strategy == 'attention':
            self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        elif strategy == 'transformer':
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embedding_dim, nhead=8, batch_first=True),
                num_layers=2
            )

    def aggregate(self, self_embedding: np.ndarray, child_embeddings: List[np.ndarray]) -> np.ndarray:

        if not child_embeddings:
            return self_embedding
        
        # type and dimension check
        if not isinstance(child_embeddings, list):
            child_embeddings = [child_embeddings]
        
        # ensure all embeddings are numpy arrays and have the same dimension
        validated_embeddings = []
        for emb in child_embeddings:
            if isinstance(emb, np.ndarray) and emb.shape == self_embedding.shape:
                validated_embeddings.append(emb)
            else:
                logger.warning(f"Skipping invalid embedding with shape {emb.shape if hasattr(emb, 'shape') else type(emb)}")
        
        if not validated_embeddings:
            return self_embedding
        
        child_embeddings = np.array(validated_embeddings)

        if self.strategy == 'mean':
            return self._mean_aggregation(self_embedding, child_embeddings)
        elif self.strategy == 'weighted_mean':
            return self._weighted_mean_aggregation(self_embedding, child_embeddings)
        elif self.strategy == 'attention':
            return self._attention_aggregation(self_embedding, child_embeddings)
        elif self.strategy == 'transformer':
            return self._transformer_aggregation(self_embedding, child_embeddings)
        else:
            return self._mean_aggregation(self_embedding, child_embeddings)
        
    def fuse_modalities(self, primary_embedding: np.ndarray, 
                       secondary_embedding: np.ndarray, 
                       alpha: float = 0.7) -> np.ndarray:
        
        return alpha * primary_embedding + (1 - alpha) * secondary_embedding
    
    def align_cross_modal(self, text_embedding: np.ndarray, 
                         image_embedding: np.ndarray,
                         alignment_strength: float = 0.1) -> tuple:
        """
        简单的跨模态对齐：通过线性变换减少语义鸿沟
        """
        # 简单的归一化对齐
        text_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)
        image_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
        
        # 计算相似度引导的对齐
        similarity = np.dot(text_norm, image_norm)
        
        # 如果相似度很低，进行轻微的投影对齐
        if similarity < 0.3:
            # 简单的投影对齐：将image向text方向偏移
            aligned_image = image_embedding + alignment_strength * (text_embedding - image_embedding)
            aligned_text = text_embedding
        else:
            aligned_image = image_embedding
            aligned_text = text_embedding
            
        return aligned_text, aligned_image
    
    def _mean_aggregation(self, self_embedding: np.ndarray, 
                         child_embeddings: List[np.ndarray]) -> np.ndarray:
        all_embeddings = [self_embedding] + child_embeddings

        return np.mean(all_embeddings, axis=0)
    
    def _weighted_mean_aggregation(self, self_embedding: np.ndarray, 
                                  child_embeddings: List[np.ndarray]) -> np.ndarray:
        weights = [2.0] + [1.0] * len(child_embeddings)
        weighted_sum = weights[0] * self_embedding
        
        for i, child_emb in enumerate(child_embeddings):
            weighted_sum += weights[i+1] * child_emb
        
        return weighted_sum / sum(weights)
    
    def _attention_aggregation(self, self_embedding: np.ndarray, 
                              child_embeddings: List[np.ndarray]) -> np.ndarray:

        # convert numpy to torch tensor
        all_embeddings = [self_embedding] + child_embeddings
        embeddings_tensor = torch.stack([torch.from_numpy(emb).float() for emb in all_embeddings])
        embeddings_tensor = embeddings_tensor.unsqueeze(0)  # Add batch dimension
        
        # use self-attention
        with torch.no_grad():
            attended, _ = self.attention(embeddings_tensor, embeddings_tensor, embeddings_tensor)
            # get the output of the first position (self_embedding position)
            result = attended[0, 0]  # [batch, seq, dim] -> [dim]
        
        return result.numpy()
    
    def _transformer_aggregation(self, self_embedding: np.ndarray, 
                                child_embeddings: List[np.ndarray]) -> np.ndarray:
        
        all_embeddings = [self_embedding] + child_embeddings
        embeddings_tensor = torch.stack([torch.from_numpy(emb).float() for emb in all_embeddings])
        embeddings_tensor = embeddings_tensor.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            transformed = self.transformer(embeddings_tensor)
            # average pooling or get the first token
            result = transformed[0].mean(dim=0)  # average pooling
        
        return result.numpy()