import torch
import torch.nn as nn
import torch.autograd as autograd
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.hidden_dim = 128
        self.seq_length = 27
        self.hidden = self.init_hidden()
        self.target_dim = 2
        self.lstm = nn.LSTM(self.seq_length, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.target_dim)

    def init_hidden(self):
        if torch.cuda.is_available():
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self,feature):
        #input (seq_len, batch, input_size)
        lstm_out, self.hidden = self.lstm(feature.view(-1,1,27),self.hidden)
        outs = []
        for time_step in range(lstm_out.size(0)):
            outs.append(self.out(lstm_out[time_step,:,:]))
        return torch.stack(outs,dim=1)


