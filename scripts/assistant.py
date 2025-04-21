import os
from dotenv import load_dotenv
import ollama
import chromadb
import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load environment variables
load_dotenv()

# Retrieve BASE_DIR from environment variables
BASE_DIR = os.getenv('BASE_DIR')

# Configuration
SYSTEM_INSTRUCTIONS_PATH = os.path.join(BASE_DIR, 'templates', 'stepbystep.txt')
PRIMER_PATH = os.path.join(BASE_DIR, 'primers', '_primer.txt')

# Initialize clients and models
client = chromadb.Client()
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')  # Using a small T5 model for summarization
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

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
    stream = ollama.chat(model='qwen2.5-coder:latest', messages=convo, stream=True)
    print('ASSISTANT: \n', end='', flush=True)

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)

    print('\n')
    original_prompt = prompt.split(' CONTEXT: ')[0].replace('USER PROMPT: ', '')
    store_conversations(original_prompt, response)
    convo.append({'role': 'assistant', 'content': response})

def create_vector_db(conversations):
    vector_db_name = 'conversations'

    try:
        vector_db = client.get_collection(name=vector_db_name)
    except chromadb.errors.InvalidCollectionException:
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

def summarize_documents(documents):
    # Concatenate documents into a single string
    text_to_summarize = " ".join(documents)
    
    # Prepare text for summarization
    inputs = t5_tokenizer.encode("summarize: " + text_to_summarize, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate summary
    summary_ids = t5_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def retrieve_and_rerank_embeddings(prompt, num_results=5):
    response = ollama.embeddings(model='nomic-embed-text', prompt=prompt)
    prompt_embedding = response['embedding']

    vector_db = client.get_collection(name='conversations')
    results = vector_db.query(
        query_embeddings=[prompt_embedding],
        n_results=num_results
    )
    
    documents = [doc for doc_list in results['documents'] for doc in doc_list]  # Flatten list of lists
    
    document_embeddings = sbert_model.encode(documents, convert_to_tensor=True)
    prompt_embedding_sbert = sbert_model.encode(prompt, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(prompt_embedding_sbert, document_embeddings)

    scored_documents = list(zip(documents, cosine_scores.tolist()[0]))
    sorted_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)
    
    flat_documents = [doc for doc, _ in sorted_documents]
    
    return flat_documents

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
context_documents = retrieve_and_rerank_embeddings(prompt=initial_prompt)
context_summary = summarize_documents(context_documents)
stream_response(prompt=f'USER PROMPT: {initial_prompt} CONTEXT: {context_summary}')

# Main interaction loop
while True:
    prompt = input('USER: \n')
    context_documents = retrieve_and_rerank_embeddings(prompt=prompt)
    context_summary = summarize_documents(context_documents)
    print("Relevant Contextual Summary:\n", context_summary)
    prompt_with_context = f'USER PROMPT: {prompt} CONTEXT: {context_summary}'
    stream_response(prompt=prompt_with_context)