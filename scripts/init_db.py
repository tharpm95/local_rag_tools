import os
from dotenv import load_dotenv
import psycopg

# Load environment variables
load_dotenv()

# Retrieve database parameters from environment variables
DB_PARAMS = {
    'dbname': os.getenv('DB_NAME', 'mydatabase'),
    'user': os.getenv('DB_USER', 'myuser'),
    'password': os.getenv('DB_PASSWORD', 'mypassword'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

# Path to your SQL file
SQL_FILE_PATH = os.path.join(os.getenv('BASE_DIR'), 'backup/postgres_backup_20250420092520.sql')

def connect_db():
    """Connect to the PostgreSQL database."""
    print("Connecting to the database...")
    conn = psycopg.connect(**DB_PARAMS)
    print("Connection established.")
    return conn

def drop_existing_objects(cursor):
    """Drop existing table and sequence if they exist."""
    print("Dropping existing table and sequence if they exist...")
    drop_commands = [
        "DROP TABLE IF EXISTS public.conversations CASCADE;",
        "DROP SEQUENCE IF EXISTS public.conversations_id_seq;"
    ]
    for command in drop_commands:
        cursor.execute(command)
    print("Existing objects dropped.")

def execute_sql_file(filepath):
    """Execute SQL commands in a file, handling structure and copy commands properly."""
    print(f"Reading SQL file from: {filepath}")
    with open(filepath, 'r') as file:
        conn = connect_db()
        cursor = conn.cursor()
        try:
            drop_existing_objects(cursor)  # Drop existing tables and sequences
            
            sql_command = ''
            for line in file:
                line = line.strip()
                if line.startswith('--') or not line:
                    continue
                if line.startswith('COPY public.conversations'):
                    print("Starting data import using COPY...")
                    with cursor.copy("COPY public.conversations (id, timestamp, prompt, response) FROM STDIN") as copy:
                        for data_line in file:
                            if data_line.strip() == '\\.':  # End of copy data
                                break
                            copy.write_row(data_line.strip().split('\t'))
                    print("Data import completed.")
                    continue
                sql_command += f' {line}'
                if sql_command.endswith(';'):
                    try:
                        cursor.execute(sql_command)
                        conn.commit()
                        print(f"Executed command: {sql_command[:50]}...")
                    except Exception as e:
                        print(f"An error occurred while executing: {sql_command[:50]}... Error: {e}")
                    sql_command = ''
        finally:
            print("Closing database connection.")
            cursor.close()
            conn.close()
            print("Connection closed.")

if __name__ == '__main__':
    execute_sql_file(SQL_FILE_PATH)