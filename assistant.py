import ollama
import chromadb

client = chromadb.Client()
message_history = [
    {
        'id': '1',
        'prompt': 'Hello, how are you?',
        'response': 'I am doing well, thank you!'
    },
    {
        'id': '2',
        'prompt': 'Hello, how aree you?',
        'response': 'I am doing weell, thank you!'
    },
    {
        'id': '3',
        'prompt': 'Hello, how areee you?',
        'response': 'I am doing weeell, thank you!'
    }
]
convo = []

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
    best_embedding = results['documents'][0][0]

    return best_embedding

create_vector_db(conversations=message_history)

while True:
    prompt = input('USER: \n')
    context = retreive_embeddings(prompt=prompt)
    prompt = f'USER PROMPT: {prompt} CONTEXT: {context}'
    stream_response(prompt=prompt)