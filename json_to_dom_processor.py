import json
import os
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import base64
import requests
import openai
from collections import defaultdict
import torch
import logging

# 设置日志
logger = logging.getLogger(__name__)

# 静态配置常量
# STATIC_QWEN2VL_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
# STATIC_QWEN2VL_DEVICE = 0

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    QWEN2VL_AVAILABLE = True
except ImportError:
    QWEN2VL_AVAILABLE = False
    logger.warning("Qwen2VL not available. Please install transformers and torch to use Qwen2VL.")


def generate_output_dir_name(dataset_name: str = "MMLongBench-Doc",
                            enable_image_description: bool = False,
                            prefer_model: str = "qwen2vl", 
                            min_image_size: int = 100,
                            max_merge_chars: int = 128) -> str:
    """生成输出目录名称
    
    格式:
    - 禁用图像描述: MMLongBench-Doc_skip-images-description
    - 启用图像描述: MMLongBench-Doc_qwen2vl-250-256
    """
    if not enable_image_description:
        prefer_model = 'skip'
        min_image_size = 0
        return f"{dataset_name}_{prefer_model}-{min_image_size}-{max_merge_chars}"
    else:
        return f"{dataset_name}_{prefer_model}-{min_image_size}-{max_merge_chars}"


class DOMNode:
    def __init__(self, tag: str, text: str = '', attrs: Optional[Dict] = None,
                 depth: int = 0, page_id: Optional[str] = None, parent: Optional['DOMNode'] = None):
        self.tag = tag
        self.text = text.strip() if text else ''
        self.attrs = attrs or {}
        self.children: List['DOMNode'] = []
        self.parent = parent
        self.depth = depth
        self.page_id = page_id
        self.metadata = {}
        self.node_type = None
        self.global_id = None
        
        self.image_context = {
            'caption': '',
            'alt_text': '',
            'title_text': '',
            'surrounding_text': '',
            'reference_texts': []
        }

    def add_child(self, child: 'DOMNode'):
        self.children.append(child)
        child.parent = self

    def assign_ids(self, prefix='node', counter=[0]):
        self.global_id = f"{prefix}_{counter[0]}"
        self.metadata['global_id'] = self.global_id
        counter[0] += 1
        for child in self.children:
            child.assign_ids(prefix, counter)

    def to_json_dict(self):
        data = {
            'tag': self.tag,
            'text': self.text if self.text else None,
            'metadata': {
                'depth': self.depth,
                'page_id': self.page_id,
                'global_id': self.global_id,
                'node_type': self.node_type,
                **self.metadata
            },
            'children': [child.to_json_dict() for child in self.children] if self.children else []
        }
        
        for k in ['class', 'src', 'href', 'data-page', 'data-id', 'style']:
            if k in self.attrs:
                data[k] = self.attrs[k]

        return {k: v for k, v in data.items() if v is not None and v != []}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DOMNode':
        """从字典数据重构 DOMNode 对象"""
        # 创建节点
        metadata = data.get('metadata', {})
        
        # 提取属性
        attrs = {}
        for k in ['class', 'src', 'href', 'data-page', 'data-id', 'style']:
            if k in data:
                attrs[k] = data[k]
        
        # 创建节点实例
        node = cls(
            tag=data['tag'],
            text=data.get('text', ''),
            attrs=attrs,
            depth=metadata.get('depth', 0),
            page_id=metadata.get('page_id'),
            parent=None
        )
        
        # 设置其他属性
        node.global_id = metadata.get('global_id')
        node.node_type = metadata.get('node_type')
        
        # 恢复完整的 metadata，过滤掉已经设置的基本字段
        node.metadata = {k: v for k, v in metadata.items() 
                        if k not in ['depth', 'page_id', 'global_id', 'node_type']}
        
        # 递归创建子节点
        children_data = data.get('children', [])
        for child_data in children_data:
            child_node = cls.from_dict(child_data)
            node.add_child(child_node)
        
        return node


class Qwen2VLService:
    """Qwen2VL视觉语言模型服务，用于图像描述"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", 
                 cuda_device: int = 0, seed: int = 42):
        if not QWEN2VL_AVAILABLE:
            raise ImportError("Qwen2VL dependencies not available. Please install transformers and torch.")
        
        self.model_name = model_name
        # self.temperature = temperature
        self.seed = seed
        self.device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
        
        # 延迟加载模型
        self.model = None
        self.processor = None
        self._model_loaded = False
        
        logger.info(f"Qwen2VL service initialized. Model will be loaded on first use.")
    
    def _load_model(self):
        """延迟加载模型"""
        if self._model_loaded:
            return
        
        logger.info(f"Loading Qwen2VL model: {self.model_name}...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": self.device}
            )
            self._model_loaded = True
            logger.info(f"Qwen2VL model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Qwen2VL model: {e}")
            raise
    
    def _prepare_image(self, image_path: str) -> Image.Image:
        """图像预处理"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path)
        return image.convert('RGB') if image.mode != 'RGB' else image
    
    
    def describe_region_in_full_page(self, full_page_image_path: str, bbox: List[float], 
                                   page_context: str = "") -> str:
        """基于整页图像和坐标区域生成描述"""
        if not self._model_loaded:
            self._load_model()
        
        try:
            # 准备整页图像
            image = self._prepare_image(full_page_image_path)
            
            # 构建包含坐标信息的提示词
            prompt = self._build_region_description_prompt(bbox, page_context)
            
            # 准备消息格式
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # 应用聊天模板
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理输入
            inputs = self.processor(
                text=text,
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # 生成参数
            generation_params = {
                "max_new_tokens": 512,
                "do_sample": False,
            }
            
            # 生成响应
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_params
                )
            
            # 解码响应
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Qwen2VL region description failed: {e}")
            return f"Region description failed: {str(e)}"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _build_region_description_prompt(self, bbox: List[float], page_context: str = "") -> str:
        """构建基于坐标区域的图像描述提示词"""
        x1, y1, x2, y2 = bbox
        
        base_prompt = f"""You are an expert document analyst. Please analyze the content within the specified rectangular region in this document page image.

**Target Region Coordinates:**
- Top-left: ({x1:.1f}, {y1:.1f})  
- Bottom-right: ({x2:.1f}, {y2:.1f})

**Instructions:**
1. Focus specifically on the content within these coordinates
2. Describe what you see in this rectangular area
3. Consider how this region relates to the surrounding page content
4. Provide a structured analysis

**Output Format:**
1. **Summary**: One sentence describing what's in the target region
2. **Content Details**: Bullet points of specific elements within the coordinates
3. **Visual Features**: Colors, layout, text, graphics within the region
4. **Context Integration**: How this region fits with the overall page layout
5. **Purpose**: The likely function or meaning of this content

Focus only on the specified coordinate region, but use the surrounding context to better understand its purpose."""
        
        if page_context.strip():
            context_prompt = f"""

**Document Context:**
{page_context[:1000]}

Use this context to better understand the purpose and meaning of the content in the target region."""
            return base_prompt + context_prompt
        
        return base_prompt
    
    
    def unload(self):
        """卸载模型释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._model_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Qwen2VL model unloaded")


class ImageDescriptionService:
    """图像描述服务，支持GPT和Qwen2VL两种模型"""
    
    def __init__(self, openai_api_key: Optional[str] = None, prefer_model: str = "qwen2vl"):
        self.openai_api_key = openai_api_key
        self.prefer_model = prefer_model
        self.qwen2vl_service = None
        
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
        
        # 只在选择 qwen2vl 时初始化 Qwen2VL 服务
        if prefer_model == "qwen2vl" and STATIC_QWEN2VL_MODEL and QWEN2VL_AVAILABLE:
            try:
                self.qwen2vl_service = Qwen2VLService(
                    model_name=STATIC_QWEN2VL_MODEL,
                    cuda_device=STATIC_QWEN2VL_DEVICE,
                    # temperature=0.0
                )
                logger.info(f"Qwen2VL service initialized with model: {STATIC_QWEN2VL_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize Qwen2VL service: {e}")
                self.qwen2vl_service = None
    
    
    def describe_region_with_qwen2vl(self, full_page_image_path: str, bbox: List[float], 
                                   page_context: str = "") -> Optional[str]:
        """使用Qwen2VL基于整页图像和坐标描述区域"""
        if not self.qwen2vl_service:
            return None
        
        try:
            return self.qwen2vl_service.describe_region_in_full_page(
                full_page_image_path, bbox, page_context
            )
        except Exception as e:
            logger.error(f"Qwen2VL区域描述失败: {e}")
            return None
    
    def describe_region_with_gpt(self, full_page_image_path: str, bbox: List[float], 
                               page_context: str = "") -> Optional[str]:
        """使用GPT基于整页图像和坐标描述区域"""
        if not self.openai_client:
            return None
        
        try:
            # 读取整页图像并转换为base64
            with open(full_page_image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # 构建包含坐标信息的描述提示词
            prompt = self._build_gpt_region_description_prompt(bbox, page_context)
            
            # 调用GPT-4 Vision API (新版本)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }],
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"GPT区域描述失败: {e}")
            return None
    
    def _build_gpt_region_description_prompt(self, bbox: List[float], page_context: str = "") -> str:
        """构建GPT基于坐标区域的图像描述提示词"""
        x1, y1, x2, y2 = bbox
        
        base_prompt = f"""Please analyze the content within the specified rectangular region in this document page image.

**Target Region Coordinates:**
- Top-left: ({x1:.1f}, {y1:.1f})  
- Bottom-right: ({x2:.1f}, {y2:.1f})

**Instructions:**
1. Focus specifically on the content within these coordinates
2. Describe what you see in this rectangular area
3. Consider how this region relates to the surrounding page content
4. Provide a structured analysis

**Output Format:**
1. **Summary**: One sentence describing what's in the target region
2. **Content Details**: Bullet points of specific elements within the coordinates
3. **Visual Features**: Colors, layout, text, graphics within the region
4. **Context Integration**: How this region fits with the overall page layout
5. **Purpose**: The likely function or meaning of this content

Focus only on the specified coordinate region, but use the surrounding context to better understand its purpose."""
        
        if page_context.strip():
            context_prompt = f"""

**Document Context:**
{page_context[:1000]}

Use this context to better understand the purpose and meaning of the content in the target region."""
            return base_prompt + context_prompt
        
        return base_prompt
    
    def describe_region(self, full_page_image_path: str, bbox: List[float], 
                       page_context: str = "") -> Optional[str]:
        """根据配置的模型描述图像区域"""
        if self.prefer_model == "gpt":
            return self.describe_region_with_gpt(full_page_image_path, bbox, page_context)
        elif self.prefer_model == "qwen2vl":
            return self.describe_region_with_qwen2vl(full_page_image_path, bbox, page_context)
        else:
            return None
    
    def cleanup(self):
        """清理资源"""
        if self.qwen2vl_service:
            self.qwen2vl_service.unload()


class JSONToDOMProcessor:
    """将JSON解析结果转换为DOM树结构的处理器"""
    
    def __init__(self, max_merge_chars: int = 128, 
                 openai_api_key: Optional[str] = None,
                 prefer_model: str = "qwen2vl",
                 enable_image_description: bool = True,
                 min_image_size: int = 50,
                 output_base_dir: str = "data/dom/MMLongBench-Doc"):
        self.max_merge_chars = max_merge_chars
        self.output_base_dir = output_base_dir
        self.prefer_model = prefer_model
        self.enable_image_description = enable_image_description
        self.min_image_size = min_image_size  # 最小图片尺寸（像素）
        self.use_full_page_method = True  # 默认使用整页+坐标方法
        
        # 初始化图像描述服务
        self.image_service = ImageDescriptionService(
            openai_api_key, prefer_model
        ) if enable_image_description else None
        
        # 存储元素索引映射
        self.element_by_index = {}
        self.image_output_dir = ""
        
        # 缓存整页图像
        self.page_images_cache = {}
        
    def process_json_file(self, json_path: str, pdf_path: Optional[str] = None) -> DOMNode:
        """
        处理单个JSON文件，转换为DOM树
        
        Args:
            json_path: JSON文件路径
            pdf_path: 对应的PDF文件路径（用于图像提取）
            
        Returns:
            DOMNode: 根节点
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if 'data' not in data or 'elements' not in data['data']:
            raise ValueError("Invalid JSON structure")
            
        document_info = data['data']['document']
        elements = data['data']['elements']
        
        # 设置输出目录
        doc_name = Path(json_path).stem
        self.image_output_dir = os.path.join(self.output_base_dir, doc_name)
        Path(self.image_output_dir).mkdir(parents=True, exist_ok=True)
        
        # 构建元素索引
        self.element_by_index = {elem['index']: elem for elem in elements}
        
        # 创建文档根节点
        root = DOMNode(
            tag='document',
            text='',
            attrs={'document_id': document_info.get('id', '')},
            depth=0,
            page_id=None
        )
        root.metadata.update({
            'document_name': document_info.get('name', ''),
            'page_count': document_info.get('page_count', 0),
            'source_type': 'pdf_json'
        })
        
        # 按照parent_chapter层次结构构建DOM树
        self._build_hierarchical_dom(root, elements, pdf_path)
        
        # 分配全局ID
        root.assign_ids(prefix=f"doc_{doc_name}")
        
        return root
    
    def _build_hierarchical_dom(self, root: DOMNode, elements: List[Dict], pdf_path: Optional[str]):
        """基于parent_chapter构建层次化DOM结构"""
        
        # 找到所有顶级元素 (parent_chapter = -1)
        root_elements = [elem for elem in elements if elem.get('parent_chapter', -1) == -1]
        
        # 为每个顶级元素创建节点并递归构建子树
        for elem in sorted(root_elements, key=lambda x: x.get('index', 0)):
            element_node = self._create_element_node(elem, pdf_path, depth=1)
            if element_node:
                root.add_child(element_node)
                # 递归添加子元素
                self._add_child_elements(element_node, elem['index'], elements, pdf_path, depth=2)
    
    def _add_child_elements(self, parent_node: DOMNode, parent_index: int, 
                           elements: List[Dict], pdf_path: Optional[str], depth: int):
        """递归添加子元素 - 改进版本"""
        
        # 找到所有子元素
        child_elements = [elem for elem in elements if elem.get('parent_chapter', -1) == parent_index]
        if not child_elements:
            return
            
        # 按index排序
        child_elements = sorted(child_elements, key=lambda x: x.get('index', 0))
        
        # 🔧 新的分组策略：识别语义相关的连续元素组
        element_groups = self._group_semantically_related_elements(child_elements)
        
        # 处理每个组
        for group in element_groups:
            if self._should_merge_elements(group):
                # 合并这个组
                merged_node = self._create_merged_text_node(group, depth)
                if merged_node:
                    parent_node.add_child(merged_node)
                    # 对于合并的节点，不需要进一步递归，因为内容已经被合并了
            else:
                # 分别处理组中的每个元素
                for elem in group:
                    element_node = self._create_element_node(elem, pdf_path, depth)
                    if element_node:
                        parent_node.add_child(element_node)
                    # 无论是否创建了节点，都要继续递归查找子元素
                    self._add_child_elements(element_node if element_node else parent_node, elem['index'], elements, pdf_path, depth + 1)
    
    def _group_semantically_related_elements(self, elements: List[Dict]) -> List[List[Dict]]:
        """🔧 将列表项分组，如果组太大则分割成多个子组"""
        if not elements:
            return []
            
        groups = []
        current_group = [elements[0]]
        
        for i in range(1, len(elements)):
            prev_elem = elements[i-1]
            curr_elem = elements[i]
            
            # 检查是否应该与前一个元素分组（语义相关）
            if self._should_group_together(prev_elem, curr_elem):
                current_group.append(curr_elem)
            else:
                # 处理当前组（如果需要，分割成多个子组）
                groups.extend(self._split_group_by_char_limit(current_group))
                current_group = [curr_elem]
        
        # 处理最后一组
        groups.extend(self._split_group_by_char_limit(current_group))
        
        return groups
    
    def _split_group_by_char_limit(self, group: List[Dict]) -> List[List[Dict]]:
        """将一个大组分割成多个符合字符限制的子组"""
        if len(group) <= 1:
            return [group]
        
        # 🔧 新策略：首先分离列表项和非列表项
        list_markers = ['•', '·', '-', '*', '○', '◦']
        list_items = []
        non_list_items = []
        
        for elem in group:
            text = elem.get('text', '').strip()
            is_list_item = (
                any(text.startswith(marker) for marker in list_markers) or 
                text.startswith(tuple('123456789'))
            )
            
            if is_list_item:
                list_items.append(elem)
            else:
                non_list_items.append(elem)
        
        result_groups = []
        
        # 非列表项保持单独
        for elem in non_list_items:
            result_groups.append([elem])
        
        # 列表项按字符限制分组
        if list_items:
            current_subgroup = []
            current_chars = 0
            
            for elem in list_items:
                elem_chars = len(elem.get('text', ''))
                
                if current_chars + elem_chars <= self.max_merge_chars and current_subgroup:
                    current_subgroup.append(elem)
                    current_chars += elem_chars
                else:
                    # 开始新的子组
                    if current_subgroup:
                        result_groups.append(current_subgroup)
                    current_subgroup = [elem]
                    current_chars = elem_chars
            
            # 添加最后一个子组
            if current_subgroup:
                result_groups.append(current_subgroup)
        
        return result_groups
    
    def _should_group_together(self, elem1: Dict, elem2: Dict) -> bool:
        """判断两个元素是否应该分组"""
        # 必须都是段落类型
        if elem1.get('type') != 'paragraph' or elem2.get('type') != 'paragraph':
            return False
            
        text1 = elem1.get('text', '').strip()
        text2 = elem2.get('text', '').strip()
        
        # 检查是否都是列表项
        list_markers = ['•', '·', '-', '*', '○', '◦']
        is_list1 = any(text1.startswith(marker) for marker in list_markers) or text1.startswith(tuple('123456789'))
        is_list2 = any(text2.startswith(marker) for marker in list_markers) or text2.startswith(tuple('123456789'))
        
        if is_list1 and is_list2:
            # 两个都是列表项，检查空间是否连续
            return self._are_elements_spatially_close(elem1, elem2)
        
        # 检查是否都是短句
        if len(text1) < 50 and len(text2) < 50:
            # 检查空间是否连续
            return self._are_elements_spatially_close(elem1, elem2)
        
        return False
    
    def _are_elements_spatially_close(self, elem1: Dict, elem2: Dict) -> bool:
        """检查两个元素在空间上是否接近"""
        outline1 = elem1.get('outline', [0, 0, 0, 0])
        outline2 = elem2.get('outline', [0, 0, 0, 0])
        
        if len(outline1) < 4 or len(outline2) < 4:
            return False
            
        # 检查垂直间距
        bottom1 = outline1[3]
        top2 = outline2[1]
        gap = abs(top2 - bottom1)
        
        # 如果间距小于50像素，认为是连续的
        return gap < 50
    
    def _should_merge_elements(self, elements: List[Dict]) -> bool:
        """判断是否应该合并元素 - 改进版本"""
        if len(elements) <= 1:
            return False
            
        # 只合并文本元素
        text_elements = [elem for elem in elements if elem.get('type') == 'paragraph']
        if len(text_elements) != len(elements):
            return False
            
        # 检查总字符数
        total_chars = sum(len(elem.get('text', '')) for elem in text_elements)
        if total_chars > self.max_merge_chars:
            return False
            
        # 检查是否都是列表项
        list_markers = ['•', '·', '-', '*', '○', '◦']
        all_list_items = True
        
        for elem in text_elements:
            text = elem.get('text', '').strip()
            is_list_item = (any(text.startswith(marker) for marker in list_markers) or 
                           text.startswith(tuple('123456789')))
            if not is_list_item:
                all_list_items = False
                break
        
        if all_list_items and len(text_elements) >= 2:
            return True
            
        # 检查是否都是短句（疑似列表项）
        avg_chars = total_chars / len(text_elements)
        if avg_chars < 30:  # 平均每个元素少于30字符，可能是列表项
            return True
            
        return False
    
    def _are_elements_sequential(self, elements: List[Dict]) -> bool:
        """检查元素是否在空间上连续"""
        if len(elements) < 2:
            return False
            
        # 按Y坐标排序
        sorted_elements = sorted(elements, key=lambda x: x.get('outline', [0, 0, 0, 0])[1])
        
        for i in range(1, len(sorted_elements)):
            prev_outline = sorted_elements[i-1].get('outline', [0, 0, 0, 0])
            curr_outline = sorted_elements[i].get('outline', [0, 0, 0, 0])
            
            if len(prev_outline) < 4 or len(curr_outline) < 4:
                continue
                
            # 检查垂直间距
            prev_bottom = prev_outline[3]
            curr_top = curr_outline[1]
            gap = curr_top - prev_bottom
            
            # 如果间距太大，认为不连续
            if gap > 50:  # 增加到50像素
                return False
                
        return True
    
    def _create_merged_text_node(self, elements: List[Dict], depth: int) -> DOMNode:
        """创建合并的文本节点"""
        texts = []
        for elem in sorted(elements, key=lambda x: x.get('index', 0)):
            text = elem.get('text', '').strip()
            if text:
                texts.append(text)
        
        # 🔧 改进的合并策略：保持列表格式
        if texts:
            # 检查是否都是列表项
            list_markers = ['•', '·', '-', '*', '○', '◦']
            all_list_items = all(any(text.startswith(marker) for marker in list_markers) or 
                               text.startswith(tuple('123456789')) for text in texts)
            
            if all_list_items:
                # 对于列表项，使用换行分隔而不是空格
                merged_text = '\n'.join(texts)
                tag = 'ul'  # 使用列表标签
                attrs = {'class': 'merged-list'}
            else:
                # 对于普通短句，使用空格分隔
                merged_text = ' '.join(texts)
                tag = 'p'
                attrs = {'class': 'merged-content'}
        else:
            merged_text = ''
            tag = 'p'
            attrs = {'class': 'merged-content'}
        
        node = DOMNode(
            tag=tag,
            text=merged_text,
            attrs=attrs,
            depth=depth
        )
        
        node.metadata.update({
            'merged_count': len(elements),
            'original_indices': [elem.get('index', -1) for elem in elements],
            'element_type': 'merged_paragraph'
        })
        
        return node
    
    def _create_element_node(self, element: Dict, pdf_path: Optional[str], depth: int) -> Optional[DOMNode]:
        """创建单个元素的DOM节点"""
        element_type = element.get('type', 'unknown')
        
        if element_type == 'paragraph':
            return self._create_paragraph_node(element, depth)
        elif element_type == 'figure':
            return self._create_figure_node(element, pdf_path, depth)
        elif element_type == 'table':
            return self._create_table_node(element, pdf_path, depth)
        elif element_type in ['page_header', 'page_footer']:
            return self._create_header_footer_node(element, element_type, depth)
        else:
            return self._create_generic_node(element, depth)
    
    def _create_paragraph_node(self, element: Dict, depth: int) -> DOMNode:
        """创建段落节点"""
        if element.get('is_chapter_title'):
            # 计算标题层级
            heading_level = self._calculate_heading_level(element)
            tag = f'h{heading_level}'
        else:
            tag = 'p'
        
        node = DOMNode(
            tag=tag,
            text=element.get('text', ''),
            attrs=self._extract_attrs(element),
            depth=depth
        )
        node.metadata.update(self._extract_metadata(element))
        node.metadata['heading_level'] = heading_level if element.get('is_chapter_title') else None
        return node
    
    def _calculate_heading_level(self, element: Dict) -> int:
        """计算标题的层级 (h1-h6)"""
        # 使用缓存避免重复计算
        if not hasattr(self, '_heading_level_cache'):
            self._heading_level_cache = {}
        
        index = element['index']
        if index in self._heading_level_cache:
            return self._heading_level_cache[index]
        
        # 追踪parent_chapter关系计算层级
        level = 1
        current_parent = element.get('parent_chapter', -1)
        checked_indices = set()
        
        while current_parent != -1 and current_parent not in checked_indices:
            checked_indices.add(current_parent)
            
            # 找到父元素
            parent_elem = self.element_by_index.get(current_parent)
            if parent_elem and parent_elem.get('is_chapter_title', False):
                level += 1
                current_parent = parent_elem.get('parent_chapter', -1)
            else:
                break
        
        # 限制在h1-h6范围内
        level = min(level, 6)
        
        # 缓存结果
        self._heading_level_cache[index] = level
        return level
    
    def _create_figure_node(self, element: Dict, pdf_path: Optional[str], depth: int) -> DOMNode:
        """创建图片节点"""
        node = DOMNode(
            tag='figure',
            text=element.get('text', ''),
            attrs=self._extract_attrs(element),
            depth=depth
        )
        node.metadata.update(self._extract_metadata(element))
        
        # 提取图片
        if pdf_path and os.path.exists(pdf_path):
            image_path = self._extract_image_from_pdf(element, pdf_path, 'figure')
            if image_path:
                # 设置src属性
                relative_path = os.path.relpath(image_path, self.output_base_dir)
                node.attrs['src'] = relative_path
                node.metadata['image_extracted'] = True
                
                # 检查是否需要生成图像描述
                should_describe = self._should_describe_image(element, image_path)
                description = None
                
                if should_describe:
                    # 获取页面上下文用于图像描述
                    page_context = self._get_page_context(element)
                    
                    # 使用整页+坐标方法
                    page_num = element.get('page', 0)
                    full_page_image = self._get_full_page_image(pdf_path, page_num)
                    
                    if full_page_image and 'outline' in element:
                        bbox = element['outline']  # [x1, y1, x2, y2]
                        description = self.image_service.describe_region(
                            full_page_image, bbox, page_context
                        )
                        node.metadata['description_method'] = f'{self.prefer_model}_full_page_region'
                    else:
                        description = None
                        node.metadata['description_method'] = 'failed_no_full_page'
                    
                    if description:
                        node.image_context['caption'] = description
                        node.metadata['ai_description'] = description
                    else:
                        node.metadata['ai_description'] = 'Failed to generate description'
                else:
                    node.metadata['ai_description'] = 'Skipped (disabled or too small)'
                
                # 创建img子节点
                img_node = DOMNode(
                    tag='img',
                    text='',
                    attrs={
                        'src': relative_path, 
                        'alt': description or element.get('text', '') or 'Image',
                        'title': element.get('text', '')
                    },
                    depth=depth + 1
                )
                node.add_child(img_node)
        
        return node
    
    def _create_table_node(self, element: Dict, pdf_path: Optional[str], depth: int) -> DOMNode:
        """创建表格节点"""
        node = DOMNode(
            tag='table',
            text='',
            attrs=self._extract_attrs(element),
            depth=depth
        )
        node.metadata.update(self._extract_metadata(element))
        
        # 处理表格内容
        cells = element.get('cells', {})
        if cells:
            self._build_table_structure(node, cells, depth + 1)
        
        # 提取表格图片
        if pdf_path and os.path.exists(pdf_path):
            table_image_path = self._extract_image_from_pdf(element, pdf_path, 'table')
            if table_image_path:
                relative_path = os.path.relpath(table_image_path, self.output_base_dir)
                node.attrs['src'] = relative_path
                node.metadata['table_image_extracted'] = True
        
        return node
    
    def _create_header_footer_node(self, element: Dict, element_type: str, depth: int) -> DOMNode:
        """创建页眉/页脚节点"""
        tag = 'header' if element_type == 'page_header' else 'footer'
        node = DOMNode(
            tag=tag,
            text=element.get('text', ''),
            attrs=self._extract_attrs(element),
            depth=depth
        )
        node.metadata.update(self._extract_metadata(element))
        return node
    
    def _create_generic_node(self, element: Dict, depth: int) -> DOMNode:
        """创建通用节点"""
        node = DOMNode(
            tag='div',
            text=element.get('text', ''),
            attrs=self._extract_attrs(element),
            depth=depth
        )
        node.metadata.update(self._extract_metadata(element))
        return node
    
    def _build_table_structure(self, table_node: DOMNode, cells: Dict, depth: int):
        """构建表格的DOM结构"""
        rows = {}
        for cell_key, cell_data in cells.items():
            if '_' in cell_key:
                row_idx, col_idx = map(int, cell_key.split('_'))
                if row_idx not in rows:
                    rows[row_idx] = {}
                rows[row_idx][col_idx] = cell_data.get('text', '')
        
        for row_idx in sorted(rows.keys()):
            tr_node = DOMNode(tag='tr', text='', depth=depth)
            
            for col_idx in sorted(rows[row_idx].keys()):
                td_node = DOMNode(
                    tag='td',
                    text=rows[row_idx][col_idx],
                    depth=depth + 1
                )
                tr_node.add_child(td_node)
                
            table_node.add_child(tr_node)
    
    def _get_page_context(self, element: Dict) -> str:
        """获取元素所在页面的上下文信息"""
        page_num = element.get('page', 0)
        page_elements = [elem for elem in self.element_by_index.values() 
                        if elem.get('page') == page_num and elem.get('type') == 'paragraph']
        
        # 提取页面上的文本内容作为上下文
        context_texts = []
        for elem in sorted(page_elements, key=lambda x: x.get('index', 0)):
            text = elem.get('text', '').strip()
            if text and len(text) > 10:  # 过滤掉太短的文本
                context_texts.append(text)
        
        return ' '.join(context_texts[:5])  # 最多取前5个文本段落
    
    def _get_full_page_image(self, pdf_path: str, page_num: int) -> Optional[str]:
        """生成并缓存整页图像"""
        cache_key = f"{pdf_path}_{page_num}"
        
        if cache_key in self.page_images_cache:
            return self.page_images_cache[cache_key]
        
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            
            # 高质量渲染整页
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            
            # 保存整页图像
            filename = f"page_{page_num+1}_full.png"
            filepath = os.path.join(self.image_output_dir, filename)
            
            pix.save(filepath)
            doc.close()
            
            # 缓存路径
            self.page_images_cache[cache_key] = filepath
            return filepath
            
        except Exception as e:
            print(f"整页图像生成失败: {e}")
            return None
    
    def _extract_image_from_pdf(self, element: Dict, pdf_path: str, prefix: str) -> Optional[str]:
        """从PDF中提取图片"""
        try:
            outline = element.get('outline', [])
            if len(outline) < 4:
                return None
                
            doc = fitz.open(pdf_path)
            page_num = element.get('page', 0)
            page = doc[page_num]
            
            x1, y1, x2, y2 = outline
            rect = fitz.Rect(x1, y1, x2, y2)
            
            # 高质量提取
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, clip=rect)
            
            # 生成文件名：页面_类型_索引.png
            element_index = element.get('index', 0)
            filename = f"page_{page_num+1}_{prefix}_{element_index}.png"
            filepath = os.path.join(self.image_output_dir, filename)
            
            pix.save(filepath)
            doc.close()
            
            return filepath
            
        except Exception as e:
            print(f"图片提取失败: {e}")
            return None
    
    def _extract_attrs(self, element: Dict) -> Dict:
        """提取元素属性"""
        attrs = {}
        
        outline = element.get('outline', [])
        if len(outline) >= 4:
            attrs['data-bbox'] = f"{outline[0]},{outline[1]},{outline[2]},{outline[3]}"
        
        if element.get('rotation', 0) != 0:
            attrs['data-rotation'] = str(element['rotation'])
            
        attrs['data-page'] = str(element.get('page', 0))
        attrs['data-index'] = str(element.get('index', -1))
            
        return attrs
    
    def _extract_metadata(self, element: Dict) -> Dict:
        """提取元素元数据"""
        metadata = {
            'original_index': element.get('index', -1),
            'parent_chapter': element.get('parent_chapter', -1),
            'is_chapter_title': element.get('is_chapter_title', False),
            'element_type': element.get('type', 'unknown'),
            'page_number': element.get('page', 0)
        }
        
        if 'outline' in element:
            metadata['bbox'] = element['outline']
            
        return metadata
    
    def _should_describe_image(self, element: Dict, image_path: str) -> bool:
        """判断是否应该为图片生成描述"""
        # 如果图像描述功能被禁用
        if not self.enable_image_description or not self.image_service:
            return False
        
        # 检查图片尺寸
        outline = element.get('outline', [])
        if len(outline) >= 4:
            width = abs(outline[2] - outline[0])
            height = abs(outline[3] - outline[1])
            
            # 如果图片太小，跳过描述
            if width < self.min_image_size or height < self.min_image_size:
                print(f"跳过小图片: {width}x{height} < {self.min_image_size}px")
                return False
        
        # 检查图片文件是否存在且有效
        if not os.path.exists(image_path):
            return False
        
        # 检查文件大小（避免空文件）
        file_size = os.path.getsize(image_path)
        if file_size < 1024:  # 小于1KB
            print(f"跳过文件过小的图片: {file_size} bytes")
            return False
        
        return True


def process_document(json_path: str, pdf_path: Optional[str] = None, 
                    openai_api_key: Optional[str] = None,
                    prefer_model: str = "qwen2vl",
                    max_merge_chars: int = 128,
                    enable_image_description: bool = True,
                    min_image_size: int = 50,
                    output_base_dir: str = "data/dom/MMLongBench-Doc") -> DOMNode:
    """处理单个文档的便捷函数"""
    processor = JSONToDOMProcessor(
        max_merge_chars=max_merge_chars,
        openai_api_key=openai_api_key,
        prefer_model=prefer_model,
        enable_image_description=enable_image_description,
        min_image_size=min_image_size,
        output_base_dir=output_base_dir
    )
    return processor.process_json_file(json_path, pdf_path)


def batch_process_documents(json_dir: str, pdf_dir: Optional[str] = None,
                           openai_api_key: Optional[str] = None,
                           prefer_model: str = "qwen2vl",
                           max_merge_chars: int = 128,
                           output_base_dir: str = "data/dom/MMLongBench-Doc") -> Dict[str, DOMNode]:
    """批量处理文档"""
    results = {}
    processor = JSONToDOMProcessor(
        max_merge_chars=max_merge_chars,
        openai_api_key=openai_api_key,
        prefer_model=prefer_model,
        output_base_dir=output_base_dir
    )
    
    json_files = Path(json_dir).glob("*.json")
    
    for json_file in json_files:
        json_path = str(json_file)
        pdf_path = None
        
        if pdf_dir:
            pdf_name = json_file.stem + ".pdf"
            potential_pdf_path = os.path.join(pdf_dir, pdf_name)
            if os.path.exists(potential_pdf_path):
                pdf_path = potential_pdf_path
        
        try:
            dom_tree = processor.process_json_file(json_path, pdf_path)
            results[json_file.stem] = dom_tree
            print(f"✅ 处理完成: {json_file.name}")
        except Exception as e:
            print(f"❌ 处理失败 {json_file.name}: {e}")
            
    return results


def process_batch_documents(json_dir: str, pdf_dir: str, base_output_dir: str,
                           mode_name: str = "批量处理",
                           enable_image_description: bool = False,
                           prefer_model: str = "qwen2vl",
                           openai_api_key: Optional[str] = None,
                           min_image_size: int = 100,
                           max_merge_chars: int = 128):
    """通用批量处理文档函数"""
    print(f"=== {mode_name}模式: 处理所有PDF文档 ===")
    
    # 生成输出目录名称
    output_dir_name = generate_output_dir_name(
        dataset_name="MMLongBench-Doc",
        enable_image_description=enable_image_description,
        prefer_model=prefer_model,
        min_image_size=min_image_size,
        max_merge_chars=max_merge_chars
    )
    output_dir = os.path.join(base_output_dir, output_dir_name)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = list(Path(json_dir).glob("*.json"))
    file_type = "测试" if "test_data" in json_dir else ""
    print(f"发现 {len(json_files)} 个{file_type}JSON文件")
    
    # 配置处理器
    processor = JSONToDOMProcessor(
        max_merge_chars=max_merge_chars,
        openai_api_key=openai_api_key,
        prefer_model=prefer_model,
        enable_image_description=enable_image_description,
        min_image_size=min_image_size,
        output_base_dir=output_dir
    )
    
    processed_count = 0
    failed_count = 0
    
    for json_file in json_files:
        try:
            print(f"\n处理: {json_file.name}")
            
            # 构建对应的PDF路径
            pdf_path = os.path.join(pdf_dir, json_file.stem + ".pdf")
            if not os.path.exists(pdf_path):
                print(f"  警告: 对应PDF文件不存在: {pdf_path}")
                pdf_path = None
            
            # 处理文档
            dom_tree = processor.process_json_file(str(json_file), pdf_path)
            
            # 保存结果
            output_path = os.path.join(output_dir, json_file.stem + ".json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dom_tree.to_json_dict(), f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ 成功保存到: {output_path}")
            processed_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            failed_count += 1
    
    print(f"\n=== {mode_name}模式处理完成 ===")
    print(f"成功处理: {processed_count} 个文档")
    print(f"处理失败: {failed_count} 个文档")
    print(f"输出目录: {output_dir}")


def process_test_documents(enable_image_description: bool = False,
                          prefer_model: str = "qwen2vl",
                          openai_api_key: Optional[str] = None,
                          min_image_size: int = 100,
                          max_merge_chars: int = 128):
    """处理test_data中的所有文档"""
    process_batch_documents(
        json_dir="test_data/dict/MMLongBench-Doc",
        pdf_dir="test_data/doc/MMLongBench-Doc",
        base_output_dir="test_data/dom",
        mode_name="测试",
        enable_image_description=enable_image_description,
        prefer_model=prefer_model,
        openai_api_key=openai_api_key,
        min_image_size=min_image_size,
        max_merge_chars=max_merge_chars
    )


def process_single_document(enable_image_description: bool = False,
                           prefer_model: str = "qwen2vl",
                           openai_api_key: Optional[str] = None,
                           min_image_size: int = 100,
                           max_merge_chars: int = 128):
    """处理单个welcome-to-nus文档"""
    json_path = "test_data/dict/MMLongBench-Doc/welcome-to-nus.json"
    pdf_path = "test_data/doc/MMLongBench-Doc/welcome-to-nus.pdf"
    
    print("=== 单文件模式: 处理 welcome-to-nus 文档 ===")
    
    # 生成输出目录名称
    output_dir_name = generate_output_dir_name(
        dataset_name="MMLongBench-Doc",
        enable_image_description=enable_image_description,
        prefer_model=prefer_model,
        min_image_size=min_image_size,
        max_merge_chars=max_merge_chars
    )
    output_base_dir = os.path.join("test_data/dom", output_dir_name)
    
    dom_tree = process_document(
        json_path,
        pdf_path,
        openai_api_key=openai_api_key,
        prefer_model=prefer_model,
        max_merge_chars=max_merge_chars,
        enable_image_description=enable_image_description,
        min_image_size=min_image_size,
        output_base_dir=output_base_dir
    )
    
    # 保存测试结果
    output_path = os.path.join(output_base_dir, "welcome-to-nus.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dom_tree.to_json_dict(), f, indent=2, ensure_ascii=False)
        
    print(f"单文件DOM树已保存到: {output_path}")
    
    # 统计信息
    def count_stats(node, stats=None):
        if stats is None:
            stats = {'headings': {}, 'merged': 0, 'images': 0, 'described': 0, 'skipped': 0}
        
        if node.tag.startswith('h'):
            stats['headings'][node.tag] = stats['headings'].get(node.tag, 0) + 1
        elif 'merged_count' in node.metadata:
            stats['merged'] += 1
        elif node.tag == 'figure':
            stats['images'] += 1
            ai_desc = node.metadata.get('ai_description', '')
            if ai_desc and ai_desc != 'Skipped (disabled or too small)':
                stats['described'] += 1
            else:
                stats['skipped'] += 1
        
        for child in node.children:
            count_stats(child, stats)
        return stats
    
    stats = count_stats(dom_tree)
    print(f"\n=== 单文件统计 ===")
    print(f"标题: {dict(sorted(stats['headings'].items()))}")
    print(f"合并节点: {stats['merged']}")
    print(f"图片总数: {stats['images']}")
    print(f"AI描述: {stats['described']}")
    print(f"跳过描述: {stats['skipped']}")


def process_all_documents(enable_image_description: bool = False,
                         prefer_model: str = "qwen2vl",
                         openai_api_key: Optional[str] = None,
                         min_image_size: int = 100,
                         max_merge_chars: int = 128):
    """处理所有文档"""
    process_batch_documents(
        json_dir="data/dict/MMLongBench-Doc",
        pdf_dir="data/doc/MMLongBench-Doc",
        base_output_dir="data/dom",
        mode_name="批量处理",
        enable_image_description=enable_image_description,
        prefer_model=prefer_model,
        openai_api_key=openai_api_key,
        min_image_size=min_image_size,
        max_merge_chars=max_merge_chars
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='JSON to DOM processor')
    parser.add_argument('--mode', choices=['test', 'batch', 'single'], default='batch', 
                       help='Processing mode: test (test_data files), batch (data files), or single (welcome-to-nus only)')
    parser.add_argument('--skip-images', action='store_true', help='Skip image descriptions completely (faster)')
    parser.add_argument('--image-model', choices=['qwen2vl', 'gpt'], default='qwen2vl', 
                       help='Model to use for image descriptions (ignored if --skip-images)')
    parser.add_argument('--openai-api-key', help='OpenAI API key for GPT model')
    parser.add_argument('--min-image-size', type=int, default=150, help='Minimum image size to describe (pixels, ignored if --skip-images)')
    parser.add_argument('--max-chars', type=int, default=512, help='Max characters for text merging')
    
    args = parser.parse_args()
    
    # 统一的图像描述逻辑
    enable_image_description = not args.skip_images
    prefer_model = args.image_model if enable_image_description else "none"
    
    # 根据选择的模型准备参数
    openai_key_param = args.openai_api_key if (enable_image_description and args.image_model == 'gpt') else None
    
    print(f"=== 处理配置 ===")
    print(f"图像描述: {'启用' if enable_image_description else '禁用'}")
    if enable_image_description:
        print(f"选择模型: {args.image_model}")
        if args.image_model == 'qwen2vl':
            print(f"描述方法: 整页图像+坐标定位")
            print(f"Qwen2VL模型: {STATIC_QWEN2VL_MODEL}")
            print(f"CUDA设备: {STATIC_QWEN2VL_DEVICE}")
        else:
            print(f"描述方法: 图像裁剪")
        print(f"最小图像尺寸: {args.min_image_size}px")
    print(f"文本合并限制: {args.max_chars} 字符")
    print()
    
    if args.mode == 'test':
        process_test_documents(
            enable_image_description=enable_image_description,
            prefer_model=prefer_model,
            openai_api_key=openai_key_param,
            min_image_size=args.min_image_size,
            max_merge_chars=args.max_chars
        )
    elif args.mode == 'batch':
        process_all_documents(
            enable_image_description=enable_image_description,
            prefer_model=prefer_model,
            openai_api_key=openai_key_param,
            min_image_size=args.min_image_size,
            max_merge_chars=args.max_chars
        )
    elif args.mode == 'single':
        process_single_document(
            enable_image_description=enable_image_description,
            prefer_model=prefer_model,
            openai_api_key=openai_key_param,
            min_image_size=args.min_image_size,
            max_merge_chars=args.max_chars
        )
    else:
        print("请指定处理模式:")
        print("  --mode test   : 处理test_data中的所有文件")
        print("  --mode batch  : 处理data中的所有文件")  
        print("  --mode single : 处理welcome-to-nus单个文件")