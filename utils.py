import torch
import random
import os
import re

def get_random_seq(file: str, file_len: int):
    seq_len = 50
    start_index = random.randint(0, file_len - seq_len)
    end_index = start_index + seq_len + 1
    return file[start_index:end_index]

def seq_to_onehot(seq: list, all_chars: str, n_chars: int):
    indices = [all_chars.index(char) for char in seq]
    tensor = torch.zeros(len(seq), 1, n_chars)
    tensor[range(len(seq)), 0, indices] = 1
    return tensor

def seq_to_index(seq: list, all_chars: str):
    indices = [all_chars.index(char) for char in seq]
    tensor = torch.tensor(indices).view(-1, 1).long()
    return tensor

def get_input_and_target(file: str, file_len: int, all_chars: list, n_chars: int):
    seq    = get_random_seq(file, file_len)
    input  = seq_to_onehot(seq[:-1], all_chars, n_chars)     
    target = seq_to_index(seq[1:], all_chars).long()
    return input, target

def save_model(model: torch.nn.Module, model_name: str = "model"):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(cur_dir, "models")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    base_name = os.path.splitext(model_name)[0]
    match = re.match(r'^(.*?)(\d+)$', base_name)
    if match:
        base_name, num = match.groups()
        num = int(num)
    else:
        num = 0
    while os.path.exists(os.path.join(folder_path, model_name+".pt")):
        num += 1
        model_name = f"{base_name}_{num}"
    model_path = os.path.join(folder_path, model_name+".pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}")