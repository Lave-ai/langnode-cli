from datatype import MessagesType
from node_type import (
    BitsAndBytesConfigNode,
    ChatTemplateNode,
    HfAutoModelForCasualLMNode,
    HfAutoTokenizerNode,
    HfModelGeneratorNode,
    OpenaiClientNode,
    OpenaiCompeletionNode,
    StringType,
    IntType,
    FloatType,
    GeminiModelNode,
    GeminiGeneratorNode
)
from pipeline_ import Pipeline
import os


def main():
    pipe_line = Pipeline(id="test")

    messages = MessagesType([{"role": "user", "content": "hello"}])

    nodes = [
        ChatTemplateNode("ch1", messages=messages),
        OpenaiClientNode(
            "op1", api_key=StringType(os.environ["LANGNODE_OPENAI_API_KEY"])
        ),
        OpenaiCompeletionNode(
            "opc1",
            model_name=StringType("gpt-4"),
            max_tokens=IntType(100),
            temperature=FloatType(0.1),
        ),
        BitsAndBytesConfigNode("bnb1", load_in=StringType("4bit")),
        HfAutoTokenizerNode(
            "tok1", base_model_id=StringType("mistralai/Mistral-7B-Instruct-v0.2")
        ),
        HfAutoModelForCasualLMNode(
            "hfmodel1", base_model_id=StringType("mistralai/Mistral-7B-Instruct-v0.2")
        ),
        HfModelGeneratorNode(
            "hfgen1",
            temperature=FloatType(0.0),
            max_new_tokens=IntType(100),
            repetition_penalty=FloatType(3.1),
        ),
        GeminiModelNode("gemodel1", model_name=StringType("gemini-pro"), api_key=StringType("")),
        GeminiGeneratorNode("gemgen1", max_output_tokens=IntType(100), temperature=FloatType(0.1))
    ]
    for node in nodes:
        pipe_line.add_node(node)

    pipe_line.add_edge(source_id="ch1", target_id="opc1", target_param_name="messages")
    pipe_line.add_edge(source_id="op1", target_id="opc1", target_param_name="client")

    pipe_line.add_edge(source_id="gemodel1", target_id="gemgen1", target_param_name="model")
    pipe_line.add_edge(source_id="ch1", target_id="gemgen1", target_param_name="messages")


    pipe_line.add_edge(
        source_id="bnb1", target_id="hfmodel1", target_param_name="quantization_config"
    )
    pipe_line.add_edge(source_id="tok1", target_id="hfgen1", target_param_name="tokenizer")
    pipe_line.add_edge(source_id="ch1", target_id="hfgen1", target_param_name="messages")
    pipe_line.add_edge(
        source_id="hfmodel1", target_id="hfgen1", target_param_name="model"
    )

    print("Visualize graph:")
    pipe_line.draw()
    print("\n\n\n")

    print("Topologically sorted nodes:")
    sorted_nodes = pipe_line.topological_sort()
    for node in sorted_nodes:
        print(node)
    print("\n\n\n")

    for node in sorted_nodes:
        print(node)
        node.run()


if __name__ == "__main__":
    main()
