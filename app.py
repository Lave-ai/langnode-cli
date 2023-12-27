import os
import torch
import typer
import warnings

from transformers import BitsAndBytesConfig
from transformers.utils.logging import set_verbosity_error, disable_progress_bar
from huggingface_hub import try_to_load_from_cache

from rich import print
from rich.console import Console
from rich.table import Table

import database as db
from entities import ModelManager, HfAutoModelForCasualLMWrapper, HfAutoTokenizerWrapper
from graph import Graph, Vertex, get_root_vertex


app = typer.Typer()
MODEL_MANAGER = None

def build(base_model_id):
    graph = Graph()

    bnb_vertex = Vertex('bnb1', BitsAndBytesConfig, "bnb_config", params={"load_in_4bit": True, 
                                                                  "bnb_4bit_use_double_quant": True, 
                                                                  "bnb_4bit_compute_dtype": torch.bfloat16})
    model_vertex = Vertex('model1', HfAutoModelForCasualLMWrapper, "model", params={"base_model_id": base_model_id})
    tokenizer_vertex = Vertex('tok1', HfAutoTokenizerWrapper, "tokenizer", params={"base_model_id": base_model_id})
    model_manager_vertex = Vertex('mmv1', ModelManager, "", params={"base_model_id": base_model_id})

    graph.add_vertex(bnb_vertex)
    graph.add_vertex(model_vertex)
    graph.add_vertex(tokenizer_vertex)
    graph.add_vertex(model_manager_vertex)

    graph.add_edge('bnb1', 'model1')
    graph.add_edge('model1', 'mmv1')
    graph.add_edge('tok1', 'mmv1')

    graph.draw()

    sorted_vertices = graph.topological_sort()
    print("Topologically sorted vertices:")
    for v in sorted_vertices:
        print(v.vertex_type, v.param_name)

    for vertex in sorted_vertices:
        print(vertex.parameters)
        built_instance = vertex.run()
        print("Built instance for vertex:", built_instance)

    root_vertex = get_root_vertex(graph)
    print("Root vertex:", root_vertex.built_instance)

    return root_vertex.built_instance

@app.command()
def run(base_model_id):
    global MODEL_MANAGER
    name = db.get_model_name_by_id(base_model_id)
    MODEL_MANAGER = build(name)
    MODEL_MANAGER.talk_to()

@app.command()
def add_model(name: str):
    db.add_model(name)
    typer.echo(f"Model added: {name}")

@app.command()
def list_models():
    rows = db.list_models()

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=12)
    table.add_column("Name", min_width=20)
    table.add_column("Cached", justify="right")

    if rows:
        for row in rows:
            filepath = try_to_load_from_cache(cache_dir=os.path.join(os.environ["HF_HOME"], "hub") , 
                                              repo_id=row[1], 
                                              filename="config.json")

            cached_status = "Yes" if isinstance(filepath, str) else "No"
            table.add_row(str(row[0]), row[1], cached_status)
    else:
        console.print("No models found.")
        return

    console.print(table)


if __name__ == "__main__":
    db.initialize_db()
    warnings.filterwarnings('ignore', category=FutureWarning)

    set_verbosity_error()
    disable_progress_bar()
    app()
