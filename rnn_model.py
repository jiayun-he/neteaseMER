import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.hidden_dim = 6
        self.target_dim = 2
        self.seq_length = 12
        self.lstm = nn.LSTM(self.seq_length,self.hidden_dim)

        self.out = nn.Linear(self.hidden_dim,self.target_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, feature):
        #input (seq_len, batch, input_size)
        lstm_out, self.hidden = self.lstm(feature.view(1,1,-1),self.hidden)
        print(lstm_out)
        target_space = self.out(lstm_out.view(-1,self.hidden_dim))
        scores = F.log_softmax(target_space,dim=1)
        #print('scores: '+ scores)
        return scores


