import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from entities import TextGenerator


class HfBitsAndBytesConfigWrapper:
    def __init__(self, load_in, **kwargs) -> None:
        self.load_in = load_in
        self.kwargs = kwargs

    def __eq__(self, other):
        if not isinstance(other, HfBitsAndBytesConfigWrapper):
            return NotImplemented
        return (self.loadin == other.load_in) and (self.kwargs == other.kwargs)

    def __repr__(self) -> str:
        return __class__.__name__.replace("Wrapper", "")

    def build(self):
        if self.load_in == "4bit":
            self.built_instance = BitsAndBytesConfig(load_in_4bit=True, **self.kwargs)
        
        elif self.load_in == "8bit":
            self.built_instance = BitsAndBytesConfig(load_in_8bit=True, **self.kwargs)

    def definition(self):
        return {
            "type": self.__repr__,
            "fields": [
                {"name": "load_in", "type": "enum", "enum_values": ["4bit", "8bit"], "required": True, "default": "4bit", "is_property": True},
            ]
        }

    @classmethod
    def from_definition(cls, load_in, **kwargs):
        return cls(load_in=load_in, **kwargs)

                            
class HfAutoModelForCasualLMWrapper:
    def __init__(self, base_model_id: str, **kwargs) -> None:
        self.base_model_id = base_model_id
        self.kwargs = kwargs
        
    def __eq__(self, other):
        if not isinstance(other, HfAutoModelForCasualLMWrapper):
            return NotImplemented
        return (self.base_model_id == other.base_model_id) and (self.kwargs == other.kwargs)

    def __repr__(self) -> str:
        return __class__.__name__.replace("Wrapper", "")

    def build(self):
        self.built_instance = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            **self.kwargs
        )

    def definition(self):
        return {
            "type": self.__repr__,
            "fields": [
                {"name": "base_model_id", "type": "str", "required": True, "default": None, "is_property": True},
                {"name": "quantization_config", "type": "HfBitsAndBytesConfig", "required": True, "default": None, "is_property": False},
            ]
        }

    @classmethod
    def from_definition(cls, base_model_id: str, **kwargs):
        special_handlers = {
            "quantization_config": lambda qc: qc.built_instance if hasattr(qc, 'built_instance') else qc
        }
        processed_kwargs = {}
        for key, value in kwargs.items():
            if key in special_handlers:
                processed_kwargs[key] = special_handlers[key](value)
        return cls(base_model_id=base_model_id, **processed_kwargs)

class HfAutoTokenizerWrapper:
    def __init__(self, base_model_id: str, **kwargs) -> None:
        self.base_model_id = base_model_id
        self.kwargs = kwargs


    def __eq__(self, other):
        if not isinstance(other, HfAutoTokenizerWrapper):
            return NotImplemented
        return (self.base_model_id == other.base_model_id) and (self.kwargs == other.kwargs)

    def __repr__(self) -> str:
        return __class__.__name__.replace("Wrapper", "")

    def build(self):
        self.built_instance = AutoTokenizer.from_pretrained(self.base_model_id, **self.kwargs)

    def definition(self):
        return {
            "type": self.__repr__,
            "fields": [
                {"name": "base_model_id", "type": "str", "required": True, "default": None, "is_property": True}
            ]
        }

    @classmethod
    def from_definition(cls, base_model_id, **kwargs):
        return cls(base_model_id=base_model_id, **kwargs)


class TextGeneratorWrapper:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer


    def __eq__(self, other):
        if not isinstance(other, TextGeneratorWrapper):
            return NotImplemented
        return (self.model == other.model) and (self.tokenizer == other.tokenizer)

    def __repr__(self) -> str:
        return __class__.__name__.replace("Wrapper", "")

    def build(self):
        self.built_instance = TextGenerator(self.model, self.tokenizer)

    def __call__(self):
        self.built_instance.talk_to()

    def definition(self):
        return {
            "type": self.__repr__,
            "fields": [
                {"name": "model", "type": "HfAutoModelForCasualLM", "required": True, "default": None, "is_property": False},
                {"name": "tokenizer", "type": "HfBitsAndBytesConfig", "required": True, "default": None, "is_property": False},
            ]
        }

    @classmethod
    def from_definition(cls, model, tokenizer):
        return cls(model.built_instance, tokenizer.built_instance)
