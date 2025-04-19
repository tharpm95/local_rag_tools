import ollama
import chromadb
import psycopg
from psycopg.rows import dict_row

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
    stream = ollama.chat(model='gemma3:1b', messages=convo, stream=True)
    print('ASSISTANT: \n', end='', flush=True)

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)

    print('\n')
    store_conversations(prompt, response)
    convo.append({'role': 'assistant', 'content': response})

def create_vector_db(conversations):
    vector_db_name = 'conversations'

    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass

    vector_db = client.create_collection(name=vector_db_name)

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

conversations = fetch_conversations()
create_vector_db(conversations=conversations)

while True:
    prompt = input('USER: \n')
    context = retreive_embeddings(prompt=prompt)
    prompt = f'USER PROMPT: {prompt} CONTEXT: {context}'
    stream_response(prompt=prompt)