import sqlite3
import typer
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel, PeftModelForCausalLM


app = typer.Typer()

MODEL = None
TOKENIZER = None


@app.command()
def load_model(base_model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    global MODEL 
    MODEL = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=True
    )

    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

def talk_to(eval_prompt):
    global MODEL
    global TOKENIZER
    
    MODEL.eval()
    model_input = TOKENIZER(eval_prompt, return_tensors="pt").to("cuda")
    streamer = TextStreamer(TOKENIZER, skip_prompt=True)

    _ = MODEL.generate(**model_input, streamer=streamer, max_new_tokens=30, repetition_penalty=1.15)

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


@app.command()
def start():
    initialize_db()
    typer.echo("Starting CLI app with SQLite file-based database")
    while True:
        command = typer.prompt("Enter command (add_model, list_models, check_current_model, load_model, or exit):")

        if command.lower() == 'exit':
            typer.echo("Exiting the application.")
            break
        elif command.lower() == 'add_model':
            name = typer.prompt("Enter name")
            add_model(name)
        elif command.lower() == 'list_models':
            list_models()
        elif command.lower() == 'load_model':
            name = typer.prompt("Enter name")
            load_model(name)
        elif command.lower() == 'talk_to':
            prompt = typer.prompt("Enter prompt")
            talk_to(prompt)


if __name__ == "__main__":
    app()
