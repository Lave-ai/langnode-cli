import sqlite3

db_file = 'my_database.db'

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

def add_model(name: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO llm_models (name) VALUES (?)', 
                   (name,))
    conn.commit()
    conn.close()

def list_models():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, name FROM llm_models')
    rows = cursor.fetchall()
    conn.close()
    return rows