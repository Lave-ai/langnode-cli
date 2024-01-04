import os
from langnode_type import OpenAIClientType
from pipeline_ import Node


class ChatTemplate(Node):
    def __init__(self, id: str, *, messages: list):
        super().__init__(id)
        self.parameters.update({"messages": messages})

    @staticmethod
    def _run(messages):
        for m in messages:
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
                        "type": "array",
                        "required": True,
                        "default": None,
                    }
                ]
            },
        }


class OpenaiClient(Node):
    def __init__(self, id: str, *, api_key: str = ""):
        super().__init__(id)
        self.parameters.update(
            {
                "api_key": api_key
                if api_key
                else os.environ.get("LANGNODE_OPENAI_API_KEY")
            }
        )

    @staticmethod
    def _run(api_key: str):
        return OpenAIClientType(api_key=api_key)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__,
            "fields": {
                "props": [
                    {
                        "name": "api_key",
                        "type": "str",
                        "required": True,
                        "default": None,
                    }
                ]
            },
        }


class OpenaiCompeletion(Node):
    def __init__(
        self, id: str, *, model_name: str, max_tokens: int, temperature: float
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
    def _run(client, messages, model_name, max_tokens, temperature) -> str:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__,
            "fields": {
                "input_params": [
                    {
                        "name": "client",
                        "type": "OpenaiClient",
                        "required": True,
                        "default": None,
                    },
                    {
                        "name": "messsages",
                        "type": "ChatTemplate",
                        "required": True,
                        "default": None,
                    },
                ],
                "props": [
                    {
                        "name": "model_name",
                        "type": "str",
                        "required": True,
                        "default": None,
                    },
                    {
                        "name": "max_tokens",
                        "type": "int",
                        "required": False,
                        "default": 1024,
                    },
                    {
                        "name": "temperature",
                        "type": "float",
                        "required": False,
                        "default": 0.0,
                    },
                ],
            },
        }
