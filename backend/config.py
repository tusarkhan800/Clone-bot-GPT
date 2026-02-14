from dataclasses import dataclass

@dataclass
class GPTConfig:
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def __post_init__(self):
        # Any additional initialization logic can go here
        pass
