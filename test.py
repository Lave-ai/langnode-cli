import os
import sqlite3
import typer
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel, PeftModelForCausalLM
from huggingface_hub import try_to_load_from_cache
from transformers.utils.logging import set_verbosity_error, disable_progress_bar
import warnings
from rich import print
from rich.console import Console
from rich.table import Table

from utils import ascii_art


app = typer.Typer()
db_file = 'my_database.db'
MODEL_MANAGER = None

class ModelManager:
    def __init__(self, base_model_id):
        print(f"[bold blue]{ascii_art}[/bold blue]")
        print(f"[bold red]loading...{base_model_id}[/bold red]")

        self.conversation_history = []

        # Configure BitsAndBytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
            use_auth_token=True
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

    def unload_model(self):
        """
        Unloads the model and tokenizer to free up resources.
        """
        # Clear model from GPU memory
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()  # Clear cached memory

        # Clear tokenizer
        if self.tokenizer is not None:
            del self.tokenizer

        # Optionally, you can also set these to None to break any references
        self.model = None
        self.tokenizer = None

    def talk_to(self):
        self.model.eval()

        while True:
            print("[bold yellow] user : [/bold yellow]")
            eval_prompt = input("")

            if eval_prompt == "!bye":
                self.unload_model()
                break
            elif eval_prompt == "!history":
                print(self.conversation_history)
            elif eval_prompt =="!window":
                window = self.conversation_history[-6:]
                if window[0]["role"] == 'assistant':
                    window = window[1:]
                print(window)
            else:
                self.conversation_history.append({"role": "user", "content": eval_prompt})
                window = self.conversation_history[-6:]
                if window[0]["role"] == 'assistant':
                    window = window[1:]

                model_input = self.tokenizer.apply_chat_template(window, return_tensors="pt").to("cuda")

                num_input_tokens = model_input.shape[1]

                streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

                print("[bold green]assistant : [/bold green]")

                generated_tokens = self.model.generate(model_input, 
                                                    streamer=streamer, 
                                                    max_new_tokens=100, 
                                                    repetition_penalty=1.15)

                output_tokens = generated_tokens[:, num_input_tokens:]
                assistant_response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": assistant_response})


def get_model_name_by_id(model_id: int) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT name FROM llm_models WHERE id = ?', (model_id,))
    row = cursor.fetchone()

    conn.close()

    if row:
        return row[0]  # Return the model name
    else:
        raise ValueError


@app.command()
def run(base_model_id):
    global MODEL_MANAGER
    name = get_model_name_by_id(base_model_id)
    MODEL_MANAGER = ModelManager(name)
    MODEL_MANAGER.talk_to()


def get_db_connection():
    return sqlite3.connect(db_file)

def initialize_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create the table only if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS llm_models (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()


@app.command()
def add_model(name: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO llm_models (name) VALUES (?)', 
                   (name,))
    conn.commit()
    conn.close()
    typer.echo(f"Model added: {name}")



@app.command()
def list_models():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, name FROM llm_models')
    rows = cursor.fetchall()
    conn.close()

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
    initialize_db()

    warnings.filterwarnings('ignore', category=FutureWarning)

    set_verbosity_error()
    # disable_progress_bar()
    app()
