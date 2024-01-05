from openai import OpenAI
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict
import torch


class DataType:
    pass


class PrimitiveType(DataType):
    pass


class ModelType(DataType):
    pass


class MessagesType(PrimitiveType):
    def __init__(self, data: List[Dict[str, str]]) -> None:
        super().__init__()
        self.data = data

    def __getattr__(self, name):
        return getattr(self.data, name)


class StringType(PrimitiveType):
    def __init__(self, data: str) -> None:
        super().__init__()
        self.data = data

    def __getattr__(self, name):
        return getattr(self.data, name)


class IntType(PrimitiveType):
    def __init__(self, data: int) -> None:
        super().__init__()
        self.data = data

    def __getattr__(self, name):
        return getattr(self.data, name)


class FloatType(PrimitiveType):
    def __init__(self, data: float) -> None:
        super().__init__()
        self.data = data

    def __getattr__(self, name):
        return getattr(self.data, name)


class OpenAIClientType(ModelType):
    def __init__(self, api_key: StringType) -> None:
        super().__init__()
        self.keys_to_compare = ["api_key"]
        self.api_key = api_key
        self.data = OpenAI(api_key=api_key.data)

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        self_dict_filtered = {key: self.__dict__[key] for key in self.keys_to_compare}
        other_dict_filtered = {key: other.__dict__[key] for key in self.keys_to_compare}

        return self_dict_filtered == other_dict_filtered


class GeminiModelType(ModelType):
    def __init__(self, model_name: StringType, api_key: StringType) -> None:
        super().__init__()
        self.keys_to_compare = ["model_name", "api_key"]
        self.model_name = model_name
        self.api_key = api_key

        genai.configure(api_key=self.api_key.data)
        self.data = genai.GenerativeModel(self.model_name.data)

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        self_dict_filtered = {key: self.__dict__[key] for key in self.keys_to_compare}
        other_dict_filtered = {key: other.__dict__[key] for key in self.keys_to_compare}

        return self_dict_filtered == other_dict_filtered


class BnBConfigType(ModelType):
    def __init__(self, load_in: StringType) -> None:
        super().__init__()
        self.keys_to_compare = ["load_in"]
        self.load_in = load_in

        if self.load_in.data == "4bit":
            self.data = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        elif self.load_in.data == "8bit":
            self.data = BitsAndBytesConfig(load_in_8bit=True)

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        self_dict_filtered = {key: self.__dict__[key] for key in self.keys_to_compare}
        other_dict_filtered = {key: other.__dict__[key] for key in self.keys_to_compare}

        return self_dict_filtered == other_dict_filtered


class HfAutoModelForCasualLMType(ModelType):
    def __init__(
        self, *, quantization_config: BnBConfigType, base_model_id: StringType
    ) -> None:
        super().__init__()
        self.keys_to_compare = ["base_model_id", "quantization_config"]
        self.base_model_id = base_model_id
        self.quantization_config = quantization_config
        self.data = AutoModelForCausalLM.from_pretrained(
            self.base_model_id.data,
            quantization_config=self.quantization_config.data,
            device_map="auto",
        )

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        self_dict_filtered = {key: self.__dict__[key] for key in self.keys_to_compare}
        other_dict_filtered = {key: other.__dict__[key] for key in self.keys_to_compare}

        return self_dict_filtered == other_dict_filtered


class HfAutoTokenizerType(ModelType):
    def __init__(self, *, base_model_id: StringType) -> None:
        super().__init__()
        self.keys_to_compare = ["base_model_id"]
        self.base_model_id = base_model_id
        self.data = AutoTokenizer.from_pretrained(self.base_model_id.data)

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        self_dict_filtered = {key: self.__dict__[key] for key in self.keys_to_compare}
        other_dict_filtered = {key: other.__dict__[key] for key in self.keys_to_compare}

        return self_dict_filtered == other_dict_filtered
