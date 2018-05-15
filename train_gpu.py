import sys,time,getopt
import pandas as pd
import numpy as np
from data_model import DataSet
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import rnn_model
reload(sys)
sys.setdefaultencoding('utf-8')

dataset = DataSet()
song_ids = dataset.get_songid_list()
num_of_songs = len(song_ids)

def getData(input_feature):
    total_length = 0
    is_first = True
    for song_id in song_ids:
        labels = dataset.concat_all(song_id)
        feats = dataset.load_song_features(song_id)
        num_of_rows = min(labels.shape[0],feats.shape[0])
        mfcc = feats.iloc[0:num_of_rows,8:21]
        chroma_vector = feats.iloc[0:num_of_rows,21:33]
        chroma_deviation = feats.iloc[0:num_of_rows,33]
        energy = feats.iloc[0:num_of_rows,1]
        energy_entropy = feats.iloc[0:num_of_rows,2]

        # mfcc + chroma
        mfcc_chroma = pd.concat([mfcc, chroma_vector], axis=1)
        mfcc_chroma = mfcc_chroma.as_matrix()

        # all
        all = pd.concat([energy, energy_entropy, mfcc, chroma_vector, chroma_deviation], axis=1)
        all = all.as_matrix()

        # mfcc
        mfcc = mfcc.as_matrix()

        # default
        feat = all

        if input_feature == 'MFCC':
            feat = mfcc
        elif input_feature == 'MFCC_CHROMA':
            feat = mfcc_chroma
        elif input_feature == 'ALL':
            feat = all

        total_length += num_of_rows

        if (is_first):
            feats_all = feat
            labels_all = labels.as_matrix()
            is_first = False

        else:
            feats_all = np.append(feats_all,feat,axis=0)
            labels_all = np.append(labels_all,labels,axis=0)

    return feats_all,labels_all,total_length


def prepare_sequence(seq):
    tensor = torch.from_numpy(seq.astype(float)).float()
    if torch.cuda.is_available():
        return autograd.Variable(tensor).cuda()
    else:
        return autograd.Variable(tensor)

def training(feats_all, labels_all, total_length, optimizer,loss_fn, epoch, input_feature, train):
    momentum = 0.9
    learning_rate = 0.01
    num_of_rows_per_song = total_length / num_of_songs
    print('num_of_rows_per_song: ', num_of_rows_per_song)
    print('total number of songs: ', num_of_songs)
    cut = feats_all.shape[0]/10
    learning_rate = 0.01
    #Split dataset for 10-folder cross-validation
    x = []
    y = []
    start = 0
    end = cut
    for i in range(10):
        x.append(feats_all[start:end,:])
        y.append(labels_all[start:end,0:2])
        start = end
        end += cut


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

    model_filename = str(optimizer) + '_' + str(loss_fn) + '_' + str(epoch) + '_' + str(input_feature)  + '.pkt'
    loss_change = [] #keep track of the loss in the training process


    if(train):
        #Cross-Validation
        for k in range(10):
            if k == 9:
                x_in = x[k]
                y_in = y[k]
            else:
                x_in = x[k+1]
                y_in = y[k+1]
            for index in range(10):
                if index != k:
                    x_in = np.append(x_in,x[k],axis=0)
                    y_in = np.append(y_in,y[k],axis=0)
            x_test = x[k]
            y_test = y[k]
            start = time.time()
            print('validation set number: ', k)
            print('training set shape =', x_in.shape, y_in.shape)
            print('testing set shape = ', x_test.shape, y_test.shape)
            for epoch in range(int(epoch)):
            #    for idx in range(test_x.shape[0]):
                model.zero_grad()
                model.hidden = model.init_hidden()
                if torch.cuda.is_available():
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
                    loss_change.append(loss.data[0])

            test_score = model(prepare_sequence(x_test))
            #used for accuracy calculation
            loss_new = (test_score - prepare_sequence(y_test)).abs()


            if torch.cuda.is_available():
                loss_nparr = loss_new.data[0].cpu().numpy()

            else:
                loss_nparr = loss_new.data[0].numpy()

            metric_arr = np.linspace(0.1, 0.5, 5)
            print('validation set:', k)
            for metric in metric_arr:
                accuracy = float(loss_nparr[loss_nparr <= metric].size) / float(loss_nparr.size)
                print('metric = ', metric, 'accuracy = ', accuracy)

            torch.save(model, model_filename)

        # print(loss_change)
        # plt.xlabel("time")
        # plt.ylabel("loss")
        # plt.plot(loss_change)
        # plt.show()

    else:
        model = torch.load(model_filename)

def test():
    optimizer = 'Adam'
    loss_fn = 'L1'
    epoch = 500
    input_feature = 'ALL'
    print('optim = ', optimizer, 'loss_fn = ', loss_fn, 'epoch = ', epoch,'input_feature = ', input_feature)
    feats_all, labels_all, total_length = getData(input_feature)
    training(feats_all, labels_all, total_length, optimizer, loss_fn, epoch, input_feature,
             True)

test()

# if __name__ == '__main__':
#     opts, args = getopt.getopt(sys.argv[1:], 'ho:l:e:i:d:', ['optimizer=', 'loss_fn=','epoch=','input_feature=','help'])
#
#     #default settings
#     optimizer = 'SGD'
#     loss_fn = 'MSE'
#     learning_rate = 0.01
#     momentum = 0.9
#     epoch = 500
#     input_feature = 'MSE'
#
#     for key,value in opts:
#         if key in ['-h', '--help']:
#            print('---LSTM Music Emotion Recognition---')
#            print('params:')
#            print('-h\thelp')
#            print('-o\toptimizer\tAdam, SGD')
#            print('-l\tloss function\tMSE,L1')
#            print('-e\tepoch')
#            print('-i\tinput\tMFCC, MFCC_CHROMA,ALL')
#            exit(0)
#
#         if key in ['-o', '--optimizer']:
#             optimizer = value
#
#         if key in ['-l','--loss_fn']:
#             loss_fn = value
#
#         if key in ['-e','--epoch']:
#             epoch = value
#
#         if key in ['-i','--input']:
#             input_feature = value
#
#
#     print('optim = ', optimizer, 'loss_fn = ', loss_fn,'epoch = ',epoch,'input_feature = ',input_feature)
#     feats_all,labels_all,total_length = getData(input_feature)
#     training(feats_all,labels_all,total_length,optimizer,loss_fn,epoch,input_feature, True)
