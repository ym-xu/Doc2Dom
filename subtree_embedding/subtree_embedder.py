from typing import List, Dict, Any
import logging
from .encoders.text_encoder import TextEncoder
from .encoders.image_encoder import ImageEncoder
from .encoders.table_encoder import TableEncoder
from .aggregator import Aggregator
from .utils import NodeTypeClassifier, StructuralPromptGenerator

logger = logging.getLogger(__name__)

class SubtreeEmbedder:
    """
    SubtreeEmbedder is a class that embeds subtrees of a DOM tree into a vector space.
    """
    def __init__(self,
                 model_name: str = "openai/clip-vit-large-patch14",
                 aggregator: str = "attention",
                 embedding_dim: int = 768,
                 data_dir: str = "./data",
                 align_embeddings: bool = False,
                 device: str = "cpu"):
        
        # 保存设备信息
        self.device = device
        
        # 加载共享的多模态模型（只加载一次）
        if "clip" in model_name.lower():
            # 直接使用CLIP
            from transformers import CLIPModel, CLIPProcessor
            import torch
            self.shared_model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
            self.shared_model = self.shared_model.to(device)
            self.shared_processor = CLIPProcessor.from_pretrained(model_name)
            self.model_name = model_name
            print(f"Successfully loaded CLIP model: {model_name} on {device}")
        else:
            try:
                from transformers import Blip2ForConditionalGeneration, Blip2Processor
                self.shared_model = Blip2ForConditionalGeneration.from_pretrained(model_name)
                self.shared_processor = Blip2Processor.from_pretrained(model_name)
                self.model_name = model_name
                print(f"Successfully loaded shared BLIP2 model: {model_name}")
            except Exception as e:
                print(f"Failed to load BLIP2 model, falling back to CLIP: {e}")
                # 如果BLIP2不可用，回退到CLIP
                from transformers import CLIPModel, CLIPProcessor
                model_name = "openai/clip-vit-large-patch14"
                self.shared_model = CLIPModel.from_pretrained(model_name)
                self.shared_processor = CLIPProcessor.from_pretrained(model_name)
                self.model_name = model_name
        
        # 初始化编码器，共享同一个模型实例
        self.text_encoder = TextEncoder(shared_model=self.shared_model, shared_processor=self.shared_processor, target_dim=embedding_dim, device=device)
        self.image_encoder = ImageEncoder(shared_model=self.shared_model, shared_processor=self.shared_processor, target_dim=embedding_dim, data_dir=data_dir, device=device)
        self.table_encoder = TableEncoder(shared_model=self.shared_model, shared_processor=self.shared_processor, target_dim=embedding_dim, device=device)

        # initialize the aggregator
        self.aggregator = Aggregator(aggregator, embedding_dim)
        
        # initialize the node type classifier
        self.node_type_classifier = NodeTypeClassifier()
        
        # save cross-modal alignment flag
        self.align_embeddings = align_embeddings
        
        # initialize the structural prompt generator
        self.prompt_generator = StructuralPromptGenerator()
        
        logger.info(f"SubtreeEmbedder initialized with shared_model: {self.model_name}, aggregator: {aggregator}, embedding_dim: {embedding_dim}")


    def encode_subtree(self, node, max_depth: int = 5, current_depth: int = 0) -> Dict[str, Any]:
        """
        recursion encode DOM Nodes

        Args:
            node: DOM Node
            max_depth: maximum depth of the subtree
            current_depth: current depth of the subtree

        Returns:
            {
                "embedding": np.ndarray,
                "metadata": Dict[str, Any],
                "structure": Dict[str, Any],
            }
        """
        try:
            # if the current depth is greater than the max depth, return the text fallback encoding
            if current_depth >= max_depth:
                # Handle both dictionary and object-based nodes for logging
                if isinstance(node, dict):
                    node_id = node.get('metadata', {}).get('global_id', 'unknown')
                else:
                    node_id = getattr(node, 'global_id', 'unknown')
                # logger.warning(f"Reached max depth {max_depth} for node {node_id}")
                return self._encode_as_text_fallback(node)
            
            # get the node type and structure prompt
            node_type = self.node_type_classifier.classify(node)
            structure_prompt = self.prompt_generator.generate_prompt(node, current_depth)
            
            # 根据节点类型选择编码策略
            return self._encode_node_by_type(node, node_type, structure_prompt, max_depth, current_depth)
        
        except Exception as e:
            # Handle both dictionary and object-based nodes for error logging
            node_id = node.get('metadata', {}).get('global_id', 'unknown') if isinstance(node, dict) else getattr(node, 'global_id', 'unknown')
            logger.error(f"Error encoding node {node_id}: {str(e)}")
            return self._encode_as_text_fallback(node)
    
    def _encode_node_by_type(self, node, node_type: str, structure_prompt: str, max_depth: int, current_depth: int) -> Dict[str, Any]:
        """
        根据节点类型选择编码策略
        """
        if node_type == 'table':
            # Table始终作为整体处理，保持完整的表格结构
            return self._encode_table_node(node, structure_prompt)
        
        elif node_type == 'image':
            # Image也是整体处理
            return self._encode_image_node(node, structure_prompt)
        
        elif self._is_leaf_node(node):
            # 其他叶子节点（主要是text）
            return self._encode_leaf_node(node, node_type, structure_prompt)
        
        else:
            # 其他内部节点（如div, section, heading等）
            return self._encode_internal_node(node, node_type, structure_prompt, max_depth, current_depth)
    
    def _is_leaf_node(self, node) -> bool:
        """
        check if the node is a leaf node
        注意：table和image不再被当作叶子节点，而是特殊节点类型
        """
        children = self._get_node_children(node)
        metadata = self._get_node_metadata(node)
        node_type = metadata.get('node_type')
        
        return (not children or 
                len(children) == 0 or
                node_type in ['text'])  # 移除了'image'，让table和image走特殊处理
    
    def _encode_table_node(self, node, structure_prompt: str) -> Dict[str, Any]:
        """
        专门处理table节点 - 获取完整的表格结构信息
        """
        
        # 新架构：直接将结构信息传给TableEncoder，一次完成融合
        embedding = self.table_encoder.encode(node, structure_prompt)
        
        return {
            "embedding": embedding,
            "metadata": self._extract_metadata(node),
            "structure_info": {
                "is_leaf": False,  # table有结构但整体处理
                "node_type": "table",
                "structure_prompt": structure_prompt,
                "processed_as": "unified_encoding",  # 标记为统一编码
                "has_children": len(getattr(node, 'children', [])) > 0
            }
        }
    
    def _encode_image_node(self, node, structure_prompt: str) -> Dict[str, Any]:
        """
        专门处理image节点 - 分离关注点架构
        """
        
        # 新架构：直接将结构信息传给ImageEncoder，一次完成融合
        embedding = self.image_encoder.encode(node, structure_prompt)
        
        return {
            "embedding": embedding,
            "metadata": self._extract_metadata(node),
            "structure_info": {
                "is_leaf": False,  # image作为特殊节点处理
                "node_type": "image",
                "structure_prompt": structure_prompt,
                "processed_as": "unified_encoding"  # 标记为统一编码
            }
        }
    
    def _encode_leaf_node(self, node, node_type: str, structure_prompt: str) -> Dict[str, Any]:
        """
        处理真正的叶子节点（主要是text节点）
        """
        node_text = self._get_node_text(node)
        enhanced_text = f"{structure_prompt}\n{node_text}"
        embedding = self.text_encoder.encode(enhanced_text)

        return {
            "embedding": embedding,
            "metadata": self._extract_metadata(node),
            "structure_info": {
                "is_leaf": True,
                "node_type": node_type,
                "structure_prompt": structure_prompt
            }
        }
    
    def _encode_internal_node(self, node, node_type: str, structure_prompt: str, max_depth: int, current_depth: int) -> Dict[str, Any]:
        """
        encode the internal node (recursion encode the children)
        """
        children = self._get_node_children(node)
        node_text = self._get_node_text(node)

        child_embeddings = []
        child_metadatas = []

        for child in children:
            child_result = self.encode_subtree(child, max_depth, current_depth + 1)
            child_embeddings.append(child_result['embedding'])
            child_metadatas.append(child_result['metadata'])
        
        self_text = f"{structure_prompt}\n{node_text}" if node_text else structure_prompt
        self_embedding = self.text_encoder.encode(self_text)

        aggregated_embedding = self.aggregator.aggregate(
            self_embedding=self_embedding,
            child_embeddings=child_embeddings
        )

        return {
            "embedding": aggregated_embedding,
            "metadata": self._extract_metadata(node, child_metadatas),
            "structure_info": {
                "is_leaf": False,
                "node_type": node_type,
                "num_children": len(child_embeddings),
                "structure_prompt": structure_prompt,
                "depth": current_depth
            }
        }
        
    def _encode_as_text_fallback(self, node) -> Dict[str, Any]:
        """
        when the depth is too deep, encode the node as text fallback
        """
        text_content = self._get_node_text(node) or str(node)
        embedding = self.text_encoder.encode(text_content)
        
        return {
            "embedding": embedding,
            "metadata": self._extract_metadata(node),
            "structure_info": {
                "is_leaf": True,
                "node_type": "text_fallback",
                "structure_prompt": "Deep node - text fallback"
            }
        }
    
    def _extract_metadata(self, node, child_metadatas: List[Dict] = None) -> Dict[str, Any]:
        # Handle dictionary-based nodes (our DOM structure)
        if isinstance(node, dict):
            metadata_dict = node.get('metadata', {})
            node_id = metadata_dict.get('global_id', 'unknown')
            element_type = metadata_dict.get('element_type', 'unknown')
            page_number = metadata_dict.get('page_number')
            text_length = len(node.get('text', '') or '')
            
            # 构建丰富的元数据，利用我们DOM结构的优势
            metadata = {
                "node_id": node_id,
                "dom_path": self._get_dom_path(node),
                "node_type": element_type,  # 使用element_type
                "page_number": page_number,
                "text_length": text_length,
                "bbox": metadata_dict.get('bbox'),  # 位置信息
                "is_chapter_title": metadata_dict.get('is_chapter_title', False),
                "heading_level": metadata_dict.get('heading_level'),
                "original_index": metadata_dict.get('original_index'),
                "parent_chapter": metadata_dict.get('parent_chapter'),
                "depth": metadata_dict.get('depth', 0)
            }
            
            # 如果是图片节点，添加图片特定信息
            if element_type == 'figure':
                metadata.update({
                    "image_extracted": metadata_dict.get('image_extracted', False),
                    "ai_description": metadata_dict.get('ai_description', ''),
                    "description_method": metadata_dict.get('description_method', ''),
                    "image_src": node.get('src', '')
                })
            
            # 如果是表格节点，添加表格特定信息
            elif element_type == 'table':
                metadata.update({
                    "table_image_extracted": metadata_dict.get('table_image_extracted', False)
                })
        else:
            # Handle DOMNode objects (original format)
            node_id = getattr(node, 'global_id', 'unknown')
            node_type = getattr(node, 'node_type', 'unknown')
            page_number = getattr(node, 'page_id', None)
            text_length = len(getattr(node, 'text', '') or '')

            metadata = {
                "node_id": node_id,
                "dom_path": self._get_dom_path(node),
                "node_type": node_type,
                "page_number": page_number,
                "text_length": text_length
            }

        # Note: 图像特定的元数据现在由ImageEncoder处理

        if child_metadatas:
            metadata["num_children"] = len(child_metadatas)
            metadata["child_types"] = list(set(m.get("node_type", "unknown") for m in child_metadatas))
        
        return metadata
    
    def _get_dom_path(self, node) -> str:
        # Handle dictionary-based nodes (our DOM structure)
        if isinstance(node, dict):
            metadata_dict = node.get('metadata', {})
            page = metadata_dict.get('page_number', '?')
            if page != '?' and page is not None:
                page += 1  # 转换为1-based索引
            
            element_type = metadata_dict.get('element_type', 'unknown')
            node_id = metadata_dict.get('global_id', 'unknown')
            tag = node.get('tag', '')
            
            # 构建更详细的路径信息
            path_parts = [f"Page {page}"]
            if tag:
                path_parts.append(f"{tag}")
            if element_type != 'unknown':
                path_parts.append(f"({element_type})")
            path_parts.append(f"[{node_id}]")
            
            return " > ".join(path_parts)
        else:
            # Handle DOMNode objects (original format)
            page = getattr(node, 'page_id', '?')
            node_type = getattr(node, 'node_type', 'unknown')
            node_id = getattr(node, 'global_id', 'unknown')
            
            return f"Page {page} > {node_type}({node_id})"
    
    def _get_node_attribute(self, node, attr_name: str, default=None):
        """
        通用方法：从节点获取属性，兼容字典和对象格式
        """
        if isinstance(node, dict):
            return node.get(attr_name, default)
        else:
            return getattr(node, attr_name, default)
    
    def _get_node_children(self, node) -> List:
        """
        通用方法：获取节点的子节点列表
        """
        return self._get_node_attribute(node, 'children', [])
    
    def _get_node_text(self, node) -> str:
        """
        通用方法：获取节点的文本内容
        """
        return self._get_node_attribute(node, 'text', '')
    
    def _get_node_metadata(self, node) -> Dict:
        """
        通用方法：获取节点的元数据
        """
        return self._get_node_attribute(node, 'metadata', {})
