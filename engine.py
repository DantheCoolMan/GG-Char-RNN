import torch
import utils
from tqdm.auto import tqdm

def train_step(net: torch.nn.Module,
               input: torch.Tensor, 
               target:torch.Tensor,
               loss_func: torch.nn.Module,
               opt: torch.optim.Optimizer,
               device: torch.device):
    net.train()
    seq_len = input.shape[0]
    net.zero_grad()           
    loss = 0                    
    hidden = net.init_hidden().to(device)  
    for t in range(seq_len): 
        output, hidden = net(input[t], hidden)
        loss += loss_func(output, target[t])

    loss.backward()            
    opt.step()              

    return loss / seq_len

def train(iters: int,
          file: str,
          file_len: int,
          all_chars:int,
          n_chars:int,
          net: torch.nn.Module,
          loss_func: torch.nn.Module,
          opt: torch.optim.Optimizer,
          device: torch.device,
          print_iters: int):
    loss_sum = 0
    losses = []
    net.to(device)
    for i in tqdm(range(iters)):
        input, target = utils.get_input_and_target(file, file_len, all_chars, n_chars)       
        input, target = input.to(device), target.to(device) 
        loss = train_step(net, input, target, loss_func, opt, device) 
        loss_sum += loss.item()                         
        with torch.inference_mode():
            if i % print_iters == print_iters - 1:
                losses.append(loss_sum / print_iters)
                loss_sum = 0
    return losses