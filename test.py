import os
import sqlite3
import typer
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel, PeftModelForCausalLM
from transformers.utils.logging import set_verbosity_error, disable_progress_bar
import warnings


app = typer.Typer()

MODEL_MANAGER = None

class ModelManager:
    def __init__(self, base_model_id):
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
            eval_prompt = typer.prompt("\nuser ")
            if eval_prompt == "!bye":
                self.unload_model()
                break
            elif eval_prompt == "!history":
                print(self.conversation_history)
            else:
                self.conversation_history.append({"role": "user", "content": eval_prompt})
                model_input = self.tokenizer.apply_chat_template(self.conversation_history[:5], return_tensors="pt").to("cuda")

                num_input_tokens = model_input.shape[1]

                streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

                generated_tokens = self.model.generate(model_input, 
                                                    streamer=streamer, 
                                                    max_new_tokens=100, 
                                                    repetition_penalty=1.15)

                output_tokens = generated_tokens[:, num_input_tokens:]
                assistant_response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": assistant_response})


@app.command()
def run(base_model_id):
    global MODEL_MANAGER
    MODEL_MANAGER = ModelManager(base_model_id)
    MODEL_MANAGER.talk_to()


# File path for the SQLite database
db_file = 'my_database.db'

# Function to create a connection to the SQLite database
def get_db_connection():
    return sqlite3.connect(db_file)

# Function to initialize the database
def initialize_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS llm_models (
        name TEXT PRIMARY KEY,
        additional_attribute TEXT
    )
    ''')
    conn.commit()
    conn.close()

@app.command()
def add_model(name: str, additional_attribute: str = ''):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO llm_models (name, additional_attribute) VALUES (?, ?)', 
                   (name, additional_attribute))
    conn.commit()
    conn.close()
    typer.echo(f"Model added: {name} with attribute: {additional_attribute}")

@app.command()
def get_model(name: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT name, additional_attribute FROM llm_models WHERE name = ?', (name,))
    row = cursor.fetchone()
    conn.close()
    if row:
        typer.echo(f"llm_model: {row[0]}, Attribute: {row[1]}")
    else:
        typer.echo("Model not found.")

@app.command()
def list_models():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT name, additional_attribute FROM llm_models')
    rows = cursor.fetchall()
    conn.close()

    if rows:
        typer.echo("List of LLM Models:")
        for row in rows:
            typer.echo(f"Name: {row[0]}, Additional Attribute: {row[1]}")
    else:
        typer.echo("No models found.")

if __name__ == "__main__":
    initialize_db()

    warnings.filterwarnings('ignore', category=FutureWarning)

    set_verbosity_error()
    disable_progress_bar()
    app()
