import json

def load_json(filename):
    with open(filename) as file:
        return json.load(file)
  
def load_text(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()