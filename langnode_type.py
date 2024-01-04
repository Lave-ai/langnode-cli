
from openai import OpenAI

class LangNodeType:
    pass


class PrimitiveType(LangNodeType):
    pass

class ModelType(LangNodeType):
    pass

class StringType(PrimitiveType):
    def __init__(self, data: str) -> None:
        super().__init__()
        self.data = data

    def __getattr__(self, name):
        return getattr(self.data, name)

    
class OpenAIClientType(ModelType):
    def __init__(self, api_key) -> None:
        super().__init__()
        self.keys_to_compare = ["api_key"]
        self.data = OpenAI(api_key=api_key)

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        self_dict_filtered = {key: self.__dict__[key] for key in self.keys_to_compare}
        other_dict_filtered = {key: other.__dict__[key] for key in self.keys_to_compare}

        return self_dict_filtered == other_dict_filtered

    