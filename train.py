import tensorflow as tf
import sys,csv
import pandas as pd
import numpy as np
from model import DataSet
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import rnn_model
reload(sys)
sys.setdefaultencoding('utf-8')

#song_df = pd.read_csv(song_acoustic_features_dir + song_id + '.wav.csv')
#dict_df = pd.read_csv('./model/word_vec_dict.csv')
#print('song_df.shape = ' + str(song_df.shape))

#comment_df = pd.read_csv('./model/comments/' + song_id + '.csv')

#print('word_df.shape = ' + str(comment_df.shape))

#get the smaller num of cols
#num_of_rows = min(comment_df.shape[0],song_df.shape[0])

#energy_data = song_df.iloc[0:num_of_rows,0]
#mfcc_data = song_df.iloc[0:num_of_rows,10:22]
#chroma_vector_data = song_df.iloc[0:num_of_rows,23:34]
#comment_data = comment_df.iloc[0:num_of_rows,1]

#T = num_of_rows #total time length

dataset = DataSet()
song_ids = dataset.get_songid_list()
#for song_id in song_ids:
song_id = '63661'
labels = dataset.concat_all(song_id)
song_feats = dataset.load_song_features(song_id)
num_of_rows = min(labels.shape[0],song_feats.shape[0])
mfcc_data = song_feats.iloc[0:num_of_rows,10:22]
time_step = mfcc_data.shape[1]
x = mfcc_data[0:num_of_rows]
y = labels[['valence_value','arousal_value']]

#print(x.shape,y.shape) #(154,12),(154,2)

test_x = x.iloc[0:30,:].as_matrix()
test_y = y.iloc[0:30,:].as_matrix()

def prepare_sequence(seq):
    tensor = torch.from_numpy(seq).float()
    return autograd.Variable(tensor)

model = rnn_model.RNN()
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.1)
test_input = prepare_sequence(x.iloc[22:23,:].as_matrix())
test_target = prepare_sequence(y.iloc[22:23, :].as_matrix())
for epoch in range(1000):
    for idx in range(test_x.shape[0]):
        model.zero_grad()
        model.hidden = model.init_hidden()
        input = prepare_sequence(test_x[idx])
        target = prepare_sequence(test_y[idx])
        scores = model(input)
        loss = loss_function(scores,target.view(-1,2))
        loss.backward()
        optimizer.step()
        #print(loss.data[0])
    if epoch%100 == 0:
        test_score = model(test_input)
        #print('epoch', epoch, 'loss: ', loss_function(test_score, test_target).data[0])
#print(dataset.vec2word(word_vec))