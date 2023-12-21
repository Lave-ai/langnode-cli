import sqlite3
import typer
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

app = typer.Typer()


@app.command()
def load_model():
    base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)


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
        CREATE TABLE IF NOT EXISTS data (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    conn.commit()
    conn.close()

@app.command()
def add_data(key: str, value: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO data (key, value) VALUES (?, ?)', (key, value))
    conn.commit()
    conn.close()
    typer.echo(f"Data added: {key} = {value}")

@app.command()
def get_data(key: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM data WHERE key = ?', (key,))
    row = cursor.fetchone()
    conn.close()
    if row:
        typer.echo(f"Data: {row[0]}")
    else:
        typer.echo("Key not found.")

@app.command()
def start():
    initialize_db()
    typer.echo("Starting CLI app with SQLite file-based database")
    while True:
        command = typer.prompt("Enter command (add, get, or exit):")

        if command.lower() == 'exit':
            typer.echo("Exiting the application.")
            break
        elif command.lower() == 'add':
            key = typer.prompt("Enter key")
            value = typer.prompt("Enter value")
            add_data(key, value)
        elif command.lower() == 'get':
            key = typer.prompt("Enter key")
            get_data(key)

if __name__ == "__main__":
    app()
