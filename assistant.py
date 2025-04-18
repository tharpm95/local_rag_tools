import ollama

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

while True:
    prompt = input('USER: \n')
    stream_response(prompt)