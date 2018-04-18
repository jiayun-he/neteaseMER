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
import getopt
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


train = False

dataset = DataSet()
song_ids = dataset.get_songid_list()
num_of_songs = len(song_ids)
total_length = 0
is_first = True
for song_id in song_ids:
    labels = dataset.concat_all(song_id)
    feats = dataset.load_song_features(song_id)
    num_of_rows = min(labels.shape[0],feats.shape[0])
    mfcc = feats.iloc[0:num_of_rows,9:21]
    mfcc_chroma = feats.iloc[0:num_of_rows,9:32]
    chroma_vector = feats.iloc[0:num_of_rows,21:33]
    chroma_deviation = feats.iloc[0:num_of_rows,33]
    energy = feats.iloc[0:num_of_rows,1]
    energy_entropy = feats.iloc[0:num_of_rows,2]

    use_feats = pd.concat([energy,energy_entropy,mfcc,chroma_vector,chroma_deviation],axis=1)

    total_length += num_of_rows
    if (is_first):
        feats_all = use_feats.as_matrix()
        labels_all = labels.as_matrix()
        is_first = False

    else:
        feats_all = np.append(feats_all,use_feats,axis=0)
        labels_all = np.append(labels_all,labels,axis=0)


def prepare_sequence(seq):
    tensor = torch.from_numpy(seq.astype(float)).float()
    return autograd.Variable(tensor).cuda()

def training(optimizer,loss_fn, learning_rate,momentum, training_size, epoch,train):
    num_of_rows_per_song = total_length / num_of_songs
    print('num_of_rows_per_song: ', num_of_rows_per_song)
    print('total number of songs: ', num_of_songs)
    number_of_songs_in_training_set = int(training_size)
    cut = num_of_rows_per_song * number_of_songs_in_training_set
    learning_rate = 0.01
    x_in = feats_all[0:cut, :]
    y_in = labels_all[0:cut, 0:2]
    print('size of training set: ', x_in.shape, y_in.shape)
    x_test = feats_all[cut:total_length, :]
    y_test = labels_all[cut:total_length, 0:2]
    print('size of test set: ', x_test.shape, y_test.shape)
    model_filename = 'all_' + str(optimizer) + '_' + str(loss_fn) + '_' + str(learning_rate) + '_'  + str(momentum) + '_' + str(training_size) + '_' + str(epoch) + '_' + 'pkt'
    loss_change = [] #keep track of the loss in the training process

    #default
    loss_function = nn.MSELoss()

    model = rnn_model.RNN()
    if loss_fn == 'MSE':
        loss_function = nn.MSELoss()
    elif loss_fn == 'L1':
        loss_function = nn.L1Loss()

    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum)
    #plt.close()
    #fig=plt.figure()
    #plt.grid(True)
    #plt.ylim(0,1)
    #plt.legend()
    if(train):
        start = time.time()
        for epoch in range(int(epoch)):
        #    for idx in range(test_x.shape[0]):
            model.zero_grad()
            model.hidden = model.init_hidden()
            model.cuda()
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
                torch.save(model,model_filename)
                loss_change.append(loss.data[0])
                #plt.cla()
                #plt.plot(score_nparr[:,1],'x-',label = 'logits')
                #plt.plot(target_nparr[:,1],'+-',label = 'targets')
                #plt.legend()
                #plt.show()
    #print(dataset.vec2word(word_vec))
    else:
        model = torch.load(model_filename)

    test_score = model(prepare_sequence(x_test))
    loss_new = (test_score - prepare_sequence(y_test)).abs()
    loss_nparr = loss_new.data[0].cpu().numpy()

    metric_arr = np.linspace(0.1,0.5,5)
    for metric in metric_arr:
        accuracy = float(loss_nparr[loss_nparr <= metric].size) / float(loss_nparr.size)
        print('metric = ', metric, 'accuracy = ', accuracy)

    print(loss_change)
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.plot(loss_change)
    plt.show()

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'ho:l:r:m:s:e:', ['optimizer=', 'loss_fn=','learning_rate=','momentum=','training_size=','epoch=','help'])

    #default settings
    optimizer = 'Adam'
    loss_fn = 'L1'
    learning_rate = 0.01
    momentum = 0.9
    training_size = 50
    epoch = 50
    train = True

    for key,value in opts:
        if key in ['-h', '--help']:
           print('---LSTM Music Emotion Recognition---')
           print('params:')
           print('-h\thelp')
           print('-o\toptimizer\tAdam, SGD, RMSprop')
           print('-l\tloss function\tMSE,L1')
           print('-s\ttraining size\t0-150')
           print('-r\tlearning_rate')
           print('-m\tmomentum')
           print('-e\tepoch')

        if key in ['-o', '--optimizer']:
            optimizer = value

        if key in ['-l','--loss_fn']:
            loss_fn = value

        if key in ['-s','--training_size']:
            training_size = value

        if key in ['-r','--learning_rate']:
            learning_rate = value

        if key in ['-e','--epoch']:
            epoch = value

        if key in ['-m','--momentum']:
            momentum = value

    print('optim = ', optimizer, 'loss_fn = ', loss_fn, 'training_size = ',training_size, 'learning_rate = ',learning_rate, 'epoch = ',epoch)
    training(optimizer,loss_fn,learning_rate,momentum, training_size,epoch,train)
