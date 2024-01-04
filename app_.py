from node_type import ChatTemplate, OpenaiClient, OpenaiCompeletion
from pipeline_ import Pipeline, Node, Edge
import os


def main():
    pipe_line = Pipeline()

    nodes = [
    ChatTemplate("ch1", messages=[{"role": "user", "content": "hello"}]),
    OpenaiClient("op1", api_key=os.environ["LANGNODE_OPENAI_API_KEY"]),
    OpenaiCompeletion("opc1", model_name="gpt-4", max_tokens=100, temperature=0.1),
    ]
    for node in nodes:
        pipe_line.add_node(node)

    pipe_line.add_edge(source_id="ch1", target_id="opc1", target_param_name="messages")
    pipe_line.add_edge(source_id="op1", target_id="opc1", target_param_name="client")
    
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


    
    