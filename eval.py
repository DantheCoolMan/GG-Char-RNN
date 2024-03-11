import torch
import utils
import string
import os
import model

def load_model(model_name: str,
               n_chars: int,
               hidden_size: int,
               num_layers: int,
               device: torch.device):
    current_directory = os.getcwd()
    models_dir = os.path.join(current_directory, "models")
    model_path = os.path.join(models_dir, model_name)
    net = model.RNN(n_chars=n_chars, hidden_size=hidden_size, num_layers=num_layers).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    return net

def eval_step(model_name: str,
              device: torch.device, 
              all_chars: str = string.printable,
              n_chars: int = len(string.printable), 
              hidden_size: int = 512,
              num_layers: int = 3,
              init_seq: str = 'W', 
              predicted_len:str = 100,
              temp: int = 1):
    
    net = load_model(model_name, n_chars=n_chars, hidden_size=hidden_size, num_layers=num_layers, device=device)
    predicted_seq = init_seq
    hidden = net.init_hidden().to(device)
    init_input = utils.seq_to_onehot(init_seq, all_chars, n_chars).to(device)
    for t in range(len(init_seq) - 1):
        _, hidden = net(init_input[t], hidden)

    input = init_input[-1]
    for t in range(predicted_len):
        output, hidden = net(input, hidden)
        predicted_index = torch.multinomial(output.data.view(-1).div(temp).exp(), 1)[0]
        predicted_char = all_chars[predicted_index]
        predicted_seq += predicted_char
        input = utils.seq_to_onehot(predicted_char, all_chars, n_chars)[0].to(device)
    return predicted_seq
