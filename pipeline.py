from copy import deepcopy
import wrapper

class Node:
    def __init__(self, id, node_type, properties: dict):
        self.id = id
        self.wrapper_class = getattr(wrapper, node_type + "Wrapper") 
        self.wrapper = None
        self.parameters = {**properties}
        # self.previous_parameters = None
        self.edges = []
    
    def __repr__(self) -> str:
        return str(self.wrapper_class)

    def forward_pass(self, value: "Wrapper"):
        for edge in self.edges:
            if edge.source == self:
                edge.target.parameters.update({edge.target_param_name: value})

    def run(self):
        # if self.wrapper is None or self.previous_parameters != self.parameters:
        self.wrapper = self.wrapper_class.from_definition(**self.parameters)
        self.wrapper.build()

        # self.previous_parameters = deepcopy(self.parameters)

        if callable(self.wrapper):
            result = self.wrapper()
            self.forward_pass(result)
        else:
            self.forward_pass(self.wrapper)

    def add_edge(self, edge):
        if edge not in self.edges:
            self.edges.append(edge)


class Edge:
    def __init__(self, source, target, target_param_name):
        self.source = source
        self.target = target
        self.target_param_name = target_param_name


class Pipeline:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

    def find_node_by_id(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def add_edge(self, source_id, target_id, target_param_name):
        source_node = self.find_node_by_id(source_id)
        target_node = self.find_node_by_id(target_id)

        if source_node and target_node:
            edge = Edge(source_node, target_node, target_param_name)
            if edge not in self.edges:
                self.edges.append(edge)
                source_node.add_edge(edge)
                target_node.add_edge(edge)

    def topological_sort_util(self, node, visited, stack):
        visited.add(node)
        for edge in node.edges:
            if edge.target not in visited:
                self.topological_sort_util(edge.target, visited, stack)
        stack.insert(0, node)  

    def topological_sort(self):
        visited = set()
        stack = []

        for node in self.nodes:
            if node not in visited:
                self.topological_sort_util(node, visited, stack)

        return stack

    def draw(self):
        print("Pipeline Visualization:")
        for edge in self.edges:
            print(f"{edge.source.wrapper_class} --> {edge.target}.{edge.target_param_name}")


def get_root_node(pipeline):
    """
    Returns the root node of the pipeline (a node with no outgoing edges).
    """
    sources = {edge.source for edge in pipeline.edges}

    if not pipeline.edges and len(pipeline.nodes) == 1:
        return pipeline.nodes[0]

    return next((node for node in pipeline.nodes if node not in sources), None)
