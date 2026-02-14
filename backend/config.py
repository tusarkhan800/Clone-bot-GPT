from dataclasses import dataclass, asdict
from typing import Dict, Any
import json


@dataclass
class GPTConfig:
    vocab_size: int = 256
    max_seq_length: int = 512
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 10
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 500
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    max_tokens: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GPTConfig":
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_str: str) -> "GPTConfig":
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)