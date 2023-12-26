import os
import typer
import warnings

from transformers.utils.logging import set_verbosity_error, disable_progress_bar
from huggingface_hub import try_to_load_from_cache

from rich import print
from rich.console import Console
from rich.table import Table

import database as db
from model_manager import ModelManager


app = typer.Typer()
MODEL_MANAGER = None


@app.command()
def run(base_model_id):
    global MODEL_MANAGER
    name = db.get_model_name_by_id(base_model_id)
    MODEL_MANAGER = ModelManager(name)
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
