import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.hidden_dim = 64
        self.target_dim = 2
        self.seq_length = 12
        self.lstm = nn.LSTM(self.seq_length,self.hidden_dim)

        self.out = nn.Linear(self.hidden_dim,self.target_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self,feature):
        #input (seq_len, batch, input_size)
        lstm_out, self.hidden = self.lstm(feature.view(-1,1,12),self.hidden)
        outs = []
        for time_step in range(lstm_out.size(0)):
            outs.append(self.out(lstm_out[time_step,:,:]))
        return torch.stack(outs,dim=1)


