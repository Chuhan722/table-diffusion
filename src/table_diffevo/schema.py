"""
属性 schema 定义

Schema 是数据表的"公开结构信息"（不是隐私），包括：
- 每个属性的名称、类型（数值/类别）
- 类别属性的合法取值集合
- 数值属性的范围（用领域常识，不暴露数据实测范围）

生成器需要这些信息来：
1. 计算记录间距离（distance.py）
2. 变异时生成合法取值（evolution.py）
3. 初始化合成表（generator.py）

## 与隐私的边界

Schema 是**可公开的**：
- 属性名（age, education）不是秘密
- 合法取值（bachelor, high_school）是常识
- 数值范围用领域常识（18-100）而非数据实测（防止泄露真实范围）

**不可公开的**是数据的分布、计数（这些由 DP 查询保护）。

## 自动生成 schema

configs/schema.yaml 由 attribute_value_meanings.csv 自动生成：
- attribute_value_meanings.csv 定义了属性的语义和取值（公开信息）
- 自动提取生成 schema.yaml（一次性操作，已完成）
"""
from typing import List, Dict, Any
from dataclasses import dataclass
import yaml


@dataclass
class AttributeBlock:
    """单个属性块的定义"""
    name: str
    type: str  # 'numeric' 或 'categorical'
    description: str

    # 类别属性字段
    values: List[str] = None

    # 数值属性字段
    range: List[float] = None  # [min, max]

    def is_numeric(self) -> bool:
        return self.type == 'numeric'

    def is_categorical(self) -> bool:
        return self.type == 'categorical'


class Schema:
    """
    数据表的 schema 定义

    包含所有属性块的信息，提供查询接口。
    """

    def __init__(self, attributes: List[AttributeBlock]):
        self.attributes = attributes
        self._name_to_block = {attr.name: attr for attr in attributes}

    def get_block(self, name: str) -> AttributeBlock:
        """根据属性名获取块定义"""
        if name not in self._name_to_block:
            raise ValueError(f"未知属性: {name}")
        return self._name_to_block[name]

    def get_numeric_blocks(self) -> List[AttributeBlock]:
        """获取所有数值块"""
        return [attr for attr in self.attributes if attr.is_numeric()]

    def get_categorical_blocks(self) -> List[AttributeBlock]:
        """获取所有类别块"""
        return [attr for attr in self.attributes if attr.is_categorical()]

    def n_blocks(self) -> int:
        """总块数"""
        return len(self.attributes)

    def attribute_names(self) -> List[str]:
        """所有属性名"""
        return [attr.name for attr in self.attributes]


def load_schema(path: str = "configs/schema.yaml") -> Schema:
    """
    从 YAML 配置文件加载 schema

    Parameters
    ----------
    path : str
        schema 配置文件路径

    Returns
    -------
    Schema
        schema 对象

    Examples
    --------
    >>> schema = load_schema("configs/schema.yaml")
    >>> schema.n_blocks()
    10
    >>> age_block = schema.get_block("age")
    >>> age_block.type
    'numeric'
    >>> age_block.range
    [18, 100]
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    attributes = []
    for attr_dict in data['attributes']:
        attributes.append(AttributeBlock(
            name=attr_dict['name'],
            type=attr_dict['type'],
            description=attr_dict.get('description', ''),
            values=attr_dict.get('values'),
            range=attr_dict.get('range')
        ))

    return Schema(attributes)
