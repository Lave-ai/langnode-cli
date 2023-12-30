from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from entities import TextGenerator
import google.generativeai as genai
from openai import OpenAI

class OpenaiCompeletionWrapper:
    def __init__(self, client, model_name, max_tokens, temperature, prompt) -> None:
        self.client = client
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt = prompt

    def __repr__(self) -> str:
        return __class__.__name__.replace("Wrapper", "")

    def __call__(self) -> Any:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": self.prompt,
                }
            ],
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content
        
    def build(self):
        pass

    def definition(self):
        return {
            "type": self.__repr__,
            "fields": [
                {"name": "client", "type": "OpenaiClient", "required": True, "default": None, "is_property": False},
                {"name": "model_name", "type": "str", "required": True, "default": None, "is_property": True},
                {"name": "max_tokens", "type": "int", "required": False, "default": 1024, "is_property": True},
                {"name": "temperature", "type": "float", "required": False, "default": 0.0, "is_property": True},
                {"name": "prompt", "type": "str", "required": True, "default": None, "is_property": True},
            ]
        }

    @classmethod
    def from_definition(cls, client, model_name, max_tokens, temperature, prompt):
        return cls(client.built_instance, model_name, max_tokens, temperature, prompt)

class OpenaiClientWrapper:
    def __init__(self, api_key) -> None:
        self.api_key = api_key

    def __repr__(self) -> str:
        return __class__.__name__.replace("Wrapper", "")

    def build(self):
        self.built_instance = OpenAI(
            api_key=self.api_key)

    def definition(self):
        return {
            "type": self.__repr__,
            "fields": [
                {"name": "api_key", "type": "str", "required": True, "default": None, "is_property": True},
            ]
        }

    @classmethod
    def from_definition(cls, api_key):
        return cls(api_key)

class GeminiGeneratorWrapper:
    def __init__(self, model, max_output_tokens, temperature, prompt) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.prompt = prompt

    def __repr__(self) -> str:
        return __class__.__name__.replace("Wrapper", "")

    def __call__(self) -> Any:
        responses = self.model.generate_content(self.prompt, 
        generation_config={
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        },)
        print(responses.text)
        return responses.text

    def build(self):
        pass

    def definition(self):
        return {
            "type": self.__repr__,
            "fields": [
                {"name": "model", "type": "GeminiModel", "required": True, "default": None, "is_property": False},
                {"name": "max_output_tokens", "type": "int", "required": False, "default": 1024, "is_property": True},
                {"name": "temperature", "type": "float", "required": False, "default": 0.0, "is_property": True},
                {"name": "prompt", "type": "str", "required": True, "default": None, "is_property": True},
            ]
        }

    @classmethod
    def from_definition(cls, model, max_output_tokens, temperature, prompt):
        return cls(model.built_instance, max_output_tokens, temperature, prompt)


class GeminiModelWrapper:
    def __init__(self, model_name, api_key) -> None:
        self.model_name = model_name
        self.api_key = api_key

    def __repr__(self) -> str:
        return __class__.__name__.replace("Wrapper", "")

    def build(self):
        genai.configure(api_key=self.api_key)
        self.built_instance = genai.GenerativeModel(self.model_name)

    def definition(self):
        return {
            "type": self.__repr__,
            "fields": [
                {"name": "model_name", "type": "str", "required": True, "default": None, "is_property": True},
                {"name": "api_key", "type": "str", "required": True, "default": None, "is_property": True},
            ]
        }

    @classmethod
    def from_definition(cls, model_name, api_key):
        return cls(model_name, api_key)


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


class HfModelGeneratorWrapper:
    def __init__(self, model, tokenizer, temperature, max_new_tokens, repetition_penalty, prompt) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.prompt = prompt


    def __eq__(self, other):
        if not isinstance(other, TextGeneratorWrapper):
            return NotImplemented
        return (self.model == other.model) and (self.tokenizer == other.tokenizer)

    def __repr__(self) -> str:
        return __class__.__name__.replace("Wrapper", "")

    def __call__(self):
        self.model.eval()
        chat = [{"role": "user", "content": self.prompt}]
        model_input = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")
        num_input_tokens = model_input.shape[1]
        generated_tokens = self.model.generate(model_input, 
                                               max_new_tokens=self.max_new_tokens, 
                                               repetition_penalty=1.15)

        output_tokens = generated_tokens[:, num_input_tokens:]
        assistant_response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        print(assistant_response)

        return assistant_response

    def build(self):
        pass

    def definition(self):
        return {
            "type": self.__repr__,
            "fields": [
                {"name": "model", "type": "HfAutoModelForCasualLM", "required": True, "default": None, "is_property": False},
                {"name": "tokenizer", "type": "HfBitsAndBytesConfig", "required": True, "default": None, "is_property": False},
                {"name": "temperature", "type": "float", "required": False, "default": 0.0, "is_property": True},
                {"name": "max_new_tokens", "type": "int", "required": False, "default": 1024, "is_property": True},
                {"name": "repetition_penalty", "type": "float", "required": False, "default": 0.0, "is_property": True},
                {"name": "prompt", "type": "str", "required": True, "default": None, "is_property": True},
            ]
        }

    @classmethod
    def from_definition(cls, model, tokenizer, temperature, max_new_tokens, repetition_penalty, prompt):
        return cls(model.built_instance, tokenizer.built_instance, temperature, max_new_tokens, repetition_penalty, prompt)

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
