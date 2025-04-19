import ollama
import chromadb
import psycopg
from psycopg.rows import dict_row

# Configuration
SYSTEM_INSTRUCTIONS_PATH = './templates/stepbystep.txt'
PRIMER_PATH = './primers/001.txt'

# Initialize client
client = chromadb.Client()
convo = []
DB_PARAMS = {
    'dbname': 'mydatabase',
    'user': 'myuser',
    'password': 'mypassword',
    'host': 'localhost',
    'port': '5432'
}

def connect_db():
    conn = psycopg.connect(**DB_PARAMS)
    return conn

def fetch_conversations():
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT * FROM conversations")
        conversations = cursor.fetchall()
    conn.close()
    return conversations

def store_conversations(prompt, response):
    # Store only the actual prompt and response
    conn = connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO conversations (prompt, response) VALUES (%s, %s)",
            (prompt, response)
        )
        conn.commit()
    conn.close()

def stream_response(prompt):
    convo.append({'role': 'user', 'content': prompt})
    response = ''
    stream = ollama.chat(model='qwen2.5-coder:latest', messages=convo, stream=True)
    print('ASSISTANT: \n', end='', flush=True)

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)

    print('\n')
    # Store the original user prompt, without context, with the response
    original_prompt = prompt.split(' CONTEXT: ')[0].replace('USER PROMPT: ', '')
    store_conversations(original_prompt, response)
    convo.append({'role': 'assistant', 'content': response})

def create_vector_db(conversations):
    vector_db_name = 'conversations'

    # Ensure collection exists
    try:
        vector_db = client.get_collection(name=vector_db_name)
    except chromadb.errors.InvalidCollectionException:
        vector_db = client.create_collection(name=vector_db_name)

    # Add data to the collection
    for c in conversations:
        serialized_convo = f'prompt: {c["prompt"]} response: {c["response"]}'
        response = ollama.embeddings(model='nomic-embed-text', prompt=serialized_convo)
        embedding = response['embedding']

        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )

def retreive_embeddings(prompt):
    response = ollama.embeddings(model='nomic-embed-text', prompt=prompt)
    prompt_embedding = response['embedding']

    vector_db = client.get_collection(name='conversations')
    results = vector_db.query(
        query_embeddings=[prompt_embedding],
        n_results=1,
    )
    best_embedding = results['documents'][0]

    return best_embedding

def load_system_instructions():
    with open(SYSTEM_INSTRUCTIONS_PATH, 'r') as file:
        instructions = file.read()
    convo.append({'role': 'system', 'content': instructions})

def load_primer():
    with open(PRIMER_PATH, 'r') as file:
        primer_content = file.read()
    return primer_content

# Load system instructions and primer
load_system_instructions()
initial_prompt = load_primer()

# Fetch and process historical conversations
conversations = fetch_conversations()
create_vector_db(conversations=conversations)

# Perform an initial interaction with the primer
context = retreive_embeddings(prompt=initial_prompt)
stream_response(prompt=f'USER PROMPT: {initial_prompt} CONTEXT: {context}')

# Main interaction loop
while True:
    prompt = input('USER: \n')
    context = retreive_embeddings(prompt=prompt)
    prompt_with_context = f'USER PROMPT: {prompt} CONTEXT: {context}'
    stream_response(prompt=prompt_with_context)