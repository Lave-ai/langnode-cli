import os
from typing import Dict, List
from datatype import (
    FloatType,
    GeminiModelType,
    HfAutoModelForCasualLMType,
    HfAutoTokenizerType,
    IntType,
    MessagesType,
    OpenAIClientType,
    StringType,
    BnBConfigType,
)
from pipeline_ import Node


class BitsAndBytesConfigNode(Node):
    def __init__(self, id: str, *, load_in: StringType):
        super().__init__(id)
        self.parameters.update({"load_in": load_in})

    @staticmethod
    def _run(load_in: StringType) -> BnBConfigType:
        return BnBConfigType(load_in=load_in)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__,
            "fields": {
                "props": [
                    {
                        "name": "load_in",
                        "type": "StringType",
                    }
                ],
                "output": {"type": "BnBConfigType"},
            },
        }


class HfAutoTokenizerNode(Node):
    def __init__(self, id: str, *, base_model_id: StringType):
        super().__init__(id)
        self.parameters.update({"base_model_id": base_model_id})

    @staticmethod
    def _run(base_model_id: StringType) -> HfAutoTokenizerType:
        return HfAutoTokenizerType(base_model_id=base_model_id)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__,
            "fields": {
                "props": [
                    {
                        "name": "base_model_id",
                        "type": "StringType",
                    }
                ],
                "output": {"type": "HfAutoTokenizerType"},
            },
        }


class HfAutoModelForCasualLMNode(Node):
    def __init__(self, id: str, *, base_model_id: StringType):
        super().__init__(id)
        self.parameters.update({"base_model_id": base_model_id})

    @staticmethod
    def _run(
        quantization_config: BnBConfigType, base_model_id: StringType
    ) -> HfAutoModelForCasualLMType:
        return HfAutoModelForCasualLMType(
            quantization_config=quantization_config, base_model_id=base_model_id
        )

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__,
            "fields": {
                "input_params": [
                    {
                        "name": "quantization_config",
                        "type": "BnBConfigType",
                    }
                ],
                "props": [
                    {
                        "name": "base_model_id",
                        "type": "StringType",
                    }
                ],
                "output": {"type": "HfAutoModelForCasualLMType"},
            },
        }


class HfModelGeneratorNode(Node):
    def __init__(
        self,
        id: str,
        *,
        temperature: FloatType,
        max_new_tokens: IntType,
        repetition_penalty: FloatType
    ) -> None:
        super().__init__(id)
        self.parameters.update(
            {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
            }
        )

    @staticmethod
    def _run(
        model: HfAutoModelForCasualLMType,
        tokenizer: HfAutoTokenizerType,
        messages: MessagesType,
        temperature: FloatType,
        repetition_penalty: FloatType,
        max_new_tokens: IntType,
    ) -> StringType:
        model.data.eval()
        model_input = tokenizer.data.apply_chat_template(
            messages.data, return_tensors="pt", add_generation_prompt=True
        ).to("cuda")
        num_input_tokens = model_input.shape[1]
        generated_tokens = model.data.generate(
            model_input,
            max_new_tokens=max_new_tokens.data,
            repetition_penalty=repetition_penalty.data,
            temperature=temperature.data,
        )

        output_tokens = generated_tokens[:, num_input_tokens:]
        assistant_response = tokenizer.data.decode(
            output_tokens[0], skip_special_tokens=True
        )

        print(assistant_response)

        return StringType(assistant_response)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__,
            "fields": {
                "input_params": [
                    {
                        "name": "model",
                        "type": "HfAutoModelForCasualLMType",
                    },
                    {
                        "name": "tokenizer",
                        "type": "HfAutoTokenizerType",
                    },
                    {
                        "name": "messsages",
                        "type": "MessagesType",
                    },
                ],
                "props": [
                    {
                        "name": "temperature",
                        "type": "FloatType",
                    },
                    {
                        "name": "max_new_tokens",
                        "type": "IntType",
                    },
                    {
                        "name": "repetition_penalty",
                        "type": "FloatType",
                    },
                ],
                "output": {"type": "StringType"},
            },
        }


class ChatTemplateNode(Node):
    def __init__(self, id: str, *, messages: MessagesType):
        super().__init__(id)
        self.parameters.update({"messages": messages})

    @staticmethod
    def _run(messages: MessagesType) -> MessagesType:
        for m in messages.data:
            assert "role" in m
            assert "content" in m
            assert m["role"] in ("user", "system", "assistant")
        return messages

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__,
            "fields": {
                "props": [
                    {
                        "name": "messages",
                        "type": "MessagesType",
                    }
                ],
                "output": {"type": "MessagesType"},
            },
        }


class GeminiModelNode(Node):
    def __init__(self, id: str, *, model_name: StringType, api_key: StringType):
        super().__init__(id)
        self.parameters.update(
            {
                "api_key": api_key
                if api_key.data
                else StringType(os.environ.get("LANGNODE_GEMINI_API_KEY")),
                "model_name": model_name,
            }
        )

    @staticmethod
    def _run(api_key: StringType, model_name: StringType) -> GeminiModelType:
        return GeminiModelType(model_name=model_name, api_key=api_key)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__,
            "fields": {
                "props": [
                    {
                        "name": "api_key",
                        "type": "StringType",
                    },
                    {
                        "name": "model_name",
                        "type": "StringType",
                    },
                ],
                "output": {"type": "GeminiModelType"},
            },
        }


class GeminiGeneratorNode(Node):
    def __init__(self, id: str, *, max_output_tokens: IntType, temperature: FloatType):
        super().__init__(id)
        self.parameters.update(
            {"max_output_tokens": max_output_tokens, "temperature": temperature}
        )

    @staticmethod
    def _run(
        model: GeminiModelType,
        messages: MessagesType,
        max_output_tokens: IntType,
        temperature: FloatType,
    ) -> StringType:
        def to_gemini_chat_components(messages: MessagesType) -> List[Dict[str, str]]:
            new_messages = []
            for p in messages.data:
                if p["role"] == "user":
                    role = "user"
                elif p["role"] == "assistant":
                    role = "model"
                else:
                    role = "user"
                new_messages.append({"role": role, "parts": [p["content"]]})
            return new_messages

        responses = model.data.generate_content(
            to_gemini_chat_components(messages=messages),
            generation_config={
                "max_output_tokens": max_output_tokens.data,
                "temperature": temperature.data,
            },
        )
        print(responses.text)
        return StringType(responses.text)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__,
            "fields": {
                "props": [
                    {
                        "name": "max_output_tokens",
                        "type": "IntType",
                    },
                    {
                        "name": "temperature",
                        "type": "FloatType",
                    },
                ],
                "input_params": [
                    {
                        "name": "model",
                        "type": "GeminiModelType",
                    },
                    {
                        "name": "messages",
                        "type": "MessagesType",
                    },
                ],
                "output": {"type": "StringType"},
            },
        }


class OpenaiClientNode(Node):
    def __init__(self, id: str, *, api_key: StringType):
        super().__init__(id)
        self.parameters.update(
            {
                "api_key": api_key
                if api_key.data
                else StringType(os.environ.get("LANGNODE_OPENAI_API_KEY"))
            }
        )

    @staticmethod
    def _run(api_key: StringType) -> OpenAIClientType:
        return OpenAIClientType(api_key=api_key)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__,
            "fields": {
                "props": [
                    {
                        "name": "api_key",
                        "type": "StringType",
                    }
                ],
                "output": {"type": "OpenAIClientType"},
            },
        }


class OpenaiCompeletionNode(Node):
    def __init__(
        self,
        id: str,
        *,
        model_name: StringType,
        max_tokens: IntType,
        temperature: FloatType
    ):
        super().__init__(id)
        self.parameters.update(
            {
                "model_name": model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

    @staticmethod
    def _run(
        client: OpenAIClientType,
        messages: MessagesType,
        model_name: StringType,
        max_tokens: IntType,
        temperature: FloatType,
    ) -> StringType:
        chat_completion = client.chat.completions.create(
            messages=messages.data,
            model=model_name.data,
            temperature=temperature.data,
            max_tokens=max_tokens.data,
        )
        print(chat_completion.choices[0].message.content)
        return StringType(chat_completion.choices[0].message.content)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__,
            "fields": {
                "input_params": [
                    {
                        "name": "client",
                        "type": "OpenAIClientType",
                    },
                    {
                        "name": "messsages",
                        "type": "MessagesType",
                    },
                ],
                "props": [
                    {
                        "name": "model_name",
                        "type": "StringType",
                    },
                    {
                        "name": "max_tokens",
                        "type": "IntType",
                    },
                    {
                        "name": "temperature",
                        "type": "FloatType",
                    },
                ],
                "output": {"type": "StringType"},
            },
        }
