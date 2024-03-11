import os
import string
import torch
import torch.nn as nn
import model, engine, utils
import argparse

parser = argparse.ArgumentParser(description="Train an RNN model on a text file.")
parser.add_argument("--file_name", default="gg_script.txt", help="Name of the text file to train on")
parser.add_argument("--iters", type=int, default=20_000, help="Number of training iterations")
parser.add_argument("--print_iters", type=int, default=100, help="Print loss every N iterations")
parser.add_argument("--num_layers", type=int, default=3, help="Number of RNN layers")
parser.add_argument("--hidden_size", type=int, default=512, help="Size of the hidden state in the RNN")
parser.add_argument("--model_name", default='model', help="Name of the model.")
args = parser.parse_args()
all_chars = string.printable
n_chars = len(all_chars)

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, args.file_name)
file = open(file_path).read()
file_len = len(file)

iters = args.iters
print_iters = args.print_iters
num_layers = args.num_layers
hidden_size = args.hidden_size

loss_func = nn.CrossEntropyLoss()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = model.RNN(n_chars=n_chars, hidden_size=hidden_size, num_layers=num_layers).to(device)
opt = torch.optim.Adam(net.parameters(), lr=0.005) 

losses = []
losses = engine.train(iters=iters,
                        file=file,
                        file_len=file_len,
                        all_chars=all_chars,
                        n_chars=n_chars,
                        net=net,
                        loss_func=loss_func,
                        opt=opt,
                        device=device,
                        print_iters=print_iters)
utils.save_model(net, model_name = args.model_name)