import tensorflow as tf
import sys,csv
import pandas as pd
import numpy as np
from data_model import DataSet
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import rnn_model
import time
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
num_of_songs = len(song_ids)
total_length = 0
is_first = True
for song_id in song_ids:
    labels = dataset.concat_all(song_id)
    feats = dataset.load_song_features(song_id)
    num_of_rows = min(labels.shape[0],feats.shape[0])
    mfcc_data = feats.iloc[0:num_of_rows,9:21]
    chroma_vector = feats.iloc[0:num_of_rows,21:32]
    chroma_deviation = feats.iloc[0:num_of_rows,33]
    energy = feats.iloc[0:num_of_rows,1]
    total_length += num_of_rows
    if (is_first):
        feats_all = mfcc_data.as_matrix()
        labels_all = labels.as_matrix()
        is_first = False

    else:
        feats_all = np.append(feats_all,mfcc_data,axis=0)
        labels_all = np.append(labels_all,labels,axis=0)


x_in = feats_all[0:2*(total_length/3),:]
y_in = labels_all[0:2*(total_length/3),0:2]
print('size of training set: ',x_in.shape,y_in.shape)
x_test = feats_all[2*(total_length)/3:total_length,:]
y_test = labels_all[2*(total_length)/3:total_length,0:2]
print('size of test set: ',x_test.shape,y_test.shape)

def prepare_sequence(seq):
    tensor = torch.from_numpy(seq.astype(float)).float()
    return autograd.Variable(tensor)

model = rnn_model.RNN()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.1)

#plt.close()
#fig=plt.figure()
#plt.grid(True)
#plt.xlabel("time")
#plt.ylabel("arousal")
#plt.ylim(0,1)
#plt.legend()
start = time.time()
for epoch in range(100):
#    for idx in range(test_x.shape[0]):
    model.zero_grad()
    model.hidden = model.init_hidden()
    input = prepare_sequence(x_in)
    target = prepare_sequence(y_in)
    #print(input.view(-1,1,12),target)
    #prediction (batch_size * time_step * size_of(output space))
    prediction = model(input)
    loss = loss_function(prediction,target.view(1,-1,2))
    loss.backward()
    optimizer.step()
    if epoch%10 == 0:
        end = time.time()
        print('epoch: ', epoch, 'loss: ', loss.data[0])
        print('time/10epoch: ', (end-start))
        start = time.time()
        #plt.cla()
        #plt.plot(score_nparr[:,1],'x-',label = 'logits')
        #plt.plot(target_nparr[:,1],'+-',label = 'targets')
        #plt.legend()
        #plt.show()
#print(dataset.vec2word(word_vec))

test_score = model(prepare_sequence(x_test))
loss_new = loss_function(test_score,prepare_sequence(y_test))
print('epoch: ', epoch, 'loss on test set: ', loss_new.data[0])