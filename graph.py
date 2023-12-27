class Vertex:
    def __init__(self, id, vertex_type, param_name, params: dict):
        self.id = id
        self.vertex_type = vertex_type
        self.param_name = param_name
        self.parameters = params
        self.built_instance = None
        self.edges = []  # List to store connected edges

    def _build(self):
        for edge in self.edges:
            if edge.source == self:
                edge.target.parameters.update({self.param_name: self.built_instance})

        return self.built_instance


    def _run(self):
        result = self.built_instance()
        for edge in self.edges:
            if edge.source == self:
                edge.target.parameters.update({self.param_name: result})

        return result
    
    def run(self):
        self.built_instance = self.vertex_type(**self.parameters)
        if callable(self.built_instance):
            print(f"{self.built_instance} is callable")
            return self._run()
        else:
            print(f"{self.built_instance} is not callable")
            return self._build()

    def add_edge(self, edge):
        if edge not in self.edges:
            self.edges.append(edge)


class Edge:
    def __init__(self, source, target):
        self.source = source
        self.target = target


class Graph:
    def __init__(self):
        self.vertices = []
        self.edges = []

    def add_vertex(self, vertex):
        if vertex not in self.vertices:
            self.vertices.append(vertex)

    def find_vertex_by_id(self, vertex_id):
        for vertex in self.vertices:
            if vertex.id == vertex_id:
                return vertex
        return None

    def add_edge(self, source_id, target_id):
        source_vertex = self.find_vertex_by_id(source_id)
        target_vertex = self.find_vertex_by_id(target_id)

        if source_vertex and target_vertex:
            edge = Edge(source_vertex, target_vertex)
            if edge not in self.edges:
                self.edges.append(edge)
                source_vertex.add_edge(edge)
                target_vertex.add_edge(edge)

    def topological_sort_util(self, vertex, visited, stack):
        visited.add(vertex)
        for edge in vertex.edges:
            if edge.target not in visited:
                self.topological_sort_util(edge.target, visited, stack)
        stack.insert(0, vertex)  # Push vertex to stack after its adjacent vertices

    def topological_sort(self):
        visited = set()
        stack = []

        # Call the recursive helper function for all vertices not yet visited
        for vertex in self.vertices:
            if vertex not in visited:
                self.topological_sort_util(vertex, visited, stack)

        return stack  # Contains the vertices in topologically sorted order

    def draw(self):
        print("Graph Visualization:")
        for edge in self.edges:
            source_name = f"{edge.source.vertex_type.__name__}({edge.source.param_name})"
            target_name = f"{edge.target.vertex_type.__name__}({edge.target.param_name})"
            print(f"{source_name} -> {target_name}")


def get_root_vertex(graph):
    """
    Returns the root vertex of the graph (a vertex with no outgoing edges).
    """
    # Create a set of all vertices that are sources in some edges
    sources = {edge.source for edge in graph.edges}

    # If there are no edges and only one vertex, return that vertex
    if not graph.edges and len(graph.vertices) == 1:
        return graph.vertices[0]

    # Find and return the vertex that is not a source of any edge
    return next((vertex for vertex in graph.vertices if vertex not in sources), None)


class Brand:
    def __init__(self, name):
        self.name = name

class Model:
    def __init__(self, name) -> None:
        self.name = name

class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def __str__(self):
        return f"Car: {self.brand.name} {self.model.name}"

class Add3:
    def __init__(self, number):
        self.number = number

    def __call__(self):
        print(f"{self.number} + 3")
        return self.number + 3

class Multiply3:
    def __init__(self, added_result):
        self.added_result = added_result

    def __call__(self):
        print(f"{self.added_result} * 3")
        return self.added_result * 3

# Test Script
if __name__ == "__main__":
    # Create a graph
    graph = Graph()

    # Create vertices
    add3_vertex = Vertex("add1", Add3, "added_result", {"number": 2})
    multiply_vertex = Vertex("mul1", Multiply3, "multiply", {})

    # Add vertices to the graph
    graph.add_vertex(add3_vertex)
    graph.add_vertex(multiply_vertex)

    # Connect vertices with edges (Brand to Car)
    graph.add_edge("add1", "mul1")

    # Perform a topological sort
    sorted_vertices = graph.topological_sort()
    print("Topologically sorted vertices:")
    for v in sorted_vertices:
        print(v.vertex_type, v.param_name)

    # Identify the root vertex
    root_vertex = get_root_vertex(graph)
    print("Root vertex:", root_vertex.vertex_type, root_vertex.param_name)

    # Build the graph
    for vertex in sorted_vertices:
        print(vertex.parameters)
        result = vertex.run()
        print("result of vertex.run():", result)

    print("Root vertex:", root_vertex.built_instance)
    graph.draw()


# # Test Script
# if __name__ == "__main__":
#     # Create a graph
#     graph = Graph()

#     # Create vertices
#     brand_vertex = Vertex(Brand, "brand", {"name": "Tesla"})
#     model_vertex = Vertex(Model, "model", {"name": "model Y"})
#     car_vertex = Vertex(Car, "car", {})

#     # Add vertices to the graph
#     graph.add_vertex(brand_vertex)
#     graph.add_vertex(model_vertex)
#     graph.add_vertex(car_vertex)

#     # Connect vertices with edges (Brand to Car)
#     graph.add_edge(brand_vertex, car_vertex)
#     graph.add_edge(model_vertex, car_vertex)

#     # Perform a topological sort
#     sorted_vertices = graph.topological_sort()
#     print("Topologically sorted vertices:")
#     for v in sorted_vertices:
#         print(v.vertex_type, v.param_name)

#     # Identify the root vertex
#     root_vertex = get_root_vertex(graph)
#     print("Root vertex:", root_vertex.vertex_type, root_vertex.param_name)

#     # Build the graph
#     for vertex in sorted_vertices:
#         print(vertex.parameters)
#         built_instance = vertex.build()
#         print("Built instance for vertex:", built_instance)

#     print("Root vertex:", root_vertex.built_instance)
#     graph.draw()