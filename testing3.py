import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, render_template, request, send_file
import json
import os
import re

class MyGPT2Model(nn.Module):
    def __init__(self):
        super(MyGPT2Model, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, input_ids):
        outputs = self.gpt2(input_ids)
        return outputs.logits

def is_valid_endpoint(url):
    pattern = r'^https?://(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^/]*)*$'
    return re.match(pattern, url) is not None

def save_testcases_to_json(testcases, filename):
    with open(filename, 'w') as f:
        json.dump(testcases, f, indent=2)
'''
def trainer(prompt_text, num):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load('testcase_generator_model.pth'), strict=False)  # Load the adjusted model
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    num = int(num)  # Ensure num is an integer

    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=100, num_return_sequences=num, num_beams=5)
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Format the output into Postman test scripts
    postman_test_scripts = []
    for text in generated_texts:
        # Here, you'd need to process the text to ensure it's in the correct format
        # This is a placeholder to show where you would transform the GPT-2 output
        script = f"pm.test(\"Test case\", function () {{\n    {text}\n}});"
        postman_test_scripts.append(script)

    return postman_test_scripts[:num]

'''
def trainer(prompt_text, num):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load('testcase_generator_model.pth'), strict=False)  # Load the adjusted model
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    num = int(num)  # Ensure num is an integer

    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=100, num_return_sequences=num, num_beams=5)
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Format the output into Postman test scripts
    postman_test_scripts = []
    for text in generated_texts:
        # Post-processing to clean the generated text
        clean_text = clean_generated_text(text)

        script = f"pm.test(\"Test case\", function () {{\n    {clean_text}\n}});"
        postman_test_scripts.append(script)

    return postman_test_scripts[:num]

def clean_generated_text(text):
    # Remove unnecessary lines or comments
    text = re.sub(r'__.*__', '', text)
    
    # Ensure valid JavaScript code by wrapping text in appropriate syntax
    if not text.strip().startswith('pm.'):
        text = f'pm.response.to.have.status(200);\n    {text}'
    
    return text.strip()

app = Flask(__name__)
Port = 3011

@app.route("/", methods=['GET', 'POST'])
def startpy():
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def process():
    result = ""  # Initialize result with a default value
    generated_testcases = {
        "info": {
            "name": "Generated Test Cases",
            "description": "Test cases generated from the provided endpoint",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "item": []
    }
    
    query = request.form.get("query")
    n = request.form.get("num")
    if query:
        if is_valid_endpoint(query):
            try:
                # Generate test scripts
                result = trainer(query, n)
                for index, test_case_script in enumerate(result):
                    request_item = {
                        "name": f'Test Case {index+1}',
                        "request": {
                            "method": "POST",
                            "url": query,
                            "header": [],
                            "body": {
                                "mode": "raw",
                                "raw": test_case_script
                            },
                        },
                        "response": []
                    }
                    generated_testcases["item"].append(request_item)
            except Exception as e:
                generated_testcases["item"].append({
                    "name": "Error",
                    "request": {
                        "method": "POST",
                        "url": query,
                        "body": {
                            "mode": "raw",
                            "raw": str(e)
                        }
                    },
                    "response": []
                })
        else:
            result = "ENTER VALID ENDPOINT"
    
    save_testcases_to_json(generated_testcases, 'generated_testcase.json')
    return render_template("chat.html", result=result)

@app.route('/download')
def download_file():
    filename = 'generated_testcase.json'
    if os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    else:
        return 'File not found!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=Port)
