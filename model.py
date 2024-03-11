import torch
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, n_chars:int, hidden_size:int, num_layers:int):
        super(RNN, self).__init__()
        self.input_size  = n_chars 
        self.hidden_size = hidden_size      
        self.output_size = n_chars
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size=self.input_size, 
                          hidden_size=self.hidden_size,
                          num_layers=num_layers,
                          dropout = 0.5)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input:int, hidden:int):
        output, h_n = self.rnn(input, hidden)
        output = self.linear(output)
        return output, h_n

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.hidden_size)
    
    def num_layers(self):
        return self.num_layers