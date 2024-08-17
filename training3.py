import warnings
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore", category=FutureWarning)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the dataset
with open('dataset.json', 'r') as f:
    dataset = json.load(f)

# Extract prompts and testcases from the dataset
prompts = [item['prompt'] for item in dataset]
testcases = [item.get('testcase', []) for item in dataset]

# Tokenize prompts and testcases
tokenized_prompts = [tokenizer(prompt, return_tensors='pt', padding=True, truncation=True) for prompt in prompts]
tokenized_testcases = [[tokenizer(testcase['content'], return_tensors='pt', padding=True, truncation=True) for testcase in test] for test in testcases]

# Find the maximum sequence length
max_lengths = []
for tp, test in zip(tokenized_prompts, tokenized_testcases):
    prompt_length = len(tp['input_ids'][0]) if len(tp['input_ids'][0]) > 0 else 0
    testcase_lengths = [len(tc['input_ids'][0]) for tc in test if len(tc['input_ids'][0]) > 0]
    if testcase_lengths:
        max_lengths.append(max(prompt_length, max(testcase_lengths)))
    else:
        max_lengths.append(prompt_length)

max_length = max(max_lengths)

# Define dataset class with the correct padding
class MyGPT2TestcaseGenerator(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        testcase = item.get('testcase', [])

        tokenized_prompt = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        tokenized_testcases = [self.tokenizer(tc['content'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt') for tc in testcase]

        return tokenized_prompt, tokenized_testcases

def collate_fn(batch):
    prompts, testcases = zip(*batch)
    prompt_input_ids = [prompt['input_ids'].squeeze(0) for prompt in prompts]

    padded_prompts = pad_sequence(prompt_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    padded_testcases = []
    for test in testcases:
        if test:
            test_input_ids = [tc['input_ids'].squeeze(0) for tc in test]
            padded_testcases.append(pad_sequence(test_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id))
        else:
            padded_testcases.append(torch.tensor([]))

    return padded_prompts, padded_testcases

class MyGPT2Model(nn.Module):
    def __init__(self):
        super(MyGPT2Model, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, input_ids):
        # Clamp input_ids to the valid range
        input_ids = input_ids.clamp(max=self.gpt2.config.vocab_size - 1)
        outputs = self.gpt2(input_ids)
        return outputs.logits

if __name__ == '__main__':
    train_dataset = MyGPT2TestcaseGenerator(dataset, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, collate_fn=collate_fn)

    model = MyGPT2Model()
    # Set ignore_index to tokenizer.pad_token_id
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(10):  # Number of epochs
        total_loss = 0.0
        model.train()
        for batch in train_loader:
            prompts, testcases = batch
            prompts = prompts.to(device)
            optimizer.zero_grad()
            outputs = model(prompts)
            # Shift the inputs to the right to align them with the targets
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = prompts[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    

    torch.save(model.state_dict(), 'testcase_generator_model.pth')
