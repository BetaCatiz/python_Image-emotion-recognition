import json

with open("embedding.json", 'r', encoding='UTF-8') as f:
    data = json.loads(f.read())
    data = dict(data)
    print(data)
