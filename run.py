# coding=utf-8
from pyAudioAnalysis import audioFeatureExtraction
import sys,getopt,os
import torch,csv
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

def feature_extraction(filename):
    wavFileName = filename.replace('.mp3','.wav')
    name = filename.replace('.mp3','')
    command = 'avconv -i ' + filename + ' ' + wavFileName
    os.system(command)
    audioFeatureExtraction.mtFeatureExtractionToFile(wavFileName,2.0, 2.0, 0.050, 0.050,name,False,False)
    return name + '.npy'

def neuralMER(feature_file,model_file,plot = False):
    features = np.load(feature_file)
    mfcc_feat = features[:,9:21]
    model = torch.load(model_file)
    output = model(prepare_sequence(mfcc_feat))
    output_file = file(feature_file.replace('.csv','va_prediction.csv'),'wb')
    output_idf = csv.writer(output_file)
    if plot:
        plt.title(feature_file)
        plt.xlabel("time")
        plt.ylabel("arousal")
        plt.plot(output.data[0].numpy()[:,1])
        plt.show()


def prepare_sequence(seq):
    tensor = torch.from_numpy(seq.astype(float)).float()
    return autograd.Variable(tensor)

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:],'hf:o:l:r:m:s:e:',['file=','optimizer=','loss_fn=','learning_rate=','momentum=','training_size=','epoch=','help'])

    # default model
    optimizer = 'Adam'
    loss_fn = 'L1'
    learning_rate = 0.01
    momentum = 0.9
    training_size = 100
    epoch = 500
    filename = ''

    for key,value in opts:
        if key in ['-h','--help']:
            print('--Automatic Music Emotion Recognition Based on LSTM-RNN model--')
            print('params:')
            print('-h\thelp')
            print('-f\tFile Name')
            print('-o\toptimizer\tAdam, SGD, RMSprop')
            print('-l\tloss function\tMSE,L1')
            print('-s\ttraining size\t0-150')
            print('-r\tlearning_rate')
            print('-m\tmomentum')
            print('-e\tepoch')
            exit(0)

        if key in ['-f','--file']:
            filename = value
        if key in ['-o', '--optimizer']:
            optimizer = value
        if key in ['-l', '--loss_fn']:
            loss_fn = value
        if key in ['-s', '--training_size']:
            training_size = value
        if key in ['-r', '--learning_rate']:
            learning_rate = value
        if key in ['-e', '--epoch']:
            epoch = value
        if key in ['-m', '--momentum']:
            momentum = value

    print('optim = ', optimizer, 'loss_fn = ', loss_fn, 'training_size = ', training_size, 'learning_rate = ',learning_rate, 'epoch = ', epoch, 'filename = ',filename)
    model_filename = str(optimizer) + '_' + str(loss_fn) + '_' + str(learning_rate) + '_' + str(momentum) + '_' + str(training_size) + '_' + str(epoch) + '_' + 'pkt'
    features = feature_extraction(filename)
    neuralMER(features,model_filename,True)