# coding=utf-8
from pyAudioAnalysis import audioFeatureExtraction
import sys,getopt,os
import torch,csv
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def feature_extraction(filename):
    wavFileName = filename.replace('.mp3','.wav')
    name = filename.replace('.mp3','')
    command = 'avconv -i ' + filename + ' ' + wavFileName
    os.system(command)
    audioFeatureExtraction.mtFeatureExtractionToFile(wavFileName,2.0, 2.0, 0.050, 0.050,name,False,storeToCSV=True)
    return name + '.csv'

def neuralMER(feature_file,model_file,input_feature,plot = False):
    features = pd.read_csv(feature_file)
    mfcc_feat = features.iloc[:,9:21]
    chroma_vector = features.iloc[:,21:33]
    chroma_deviation = features.iloc[:,33]
    energy = features.iloc[:,1]
    energy_entropy = features.iloc[:,2]

    #mfcc + chroma
    mfcc_chroma = pd.concat([mfcc_feat,chroma_vector],axis=1)
    mfcc_chroma = mfcc_chroma.as_matrix()

    #all
    all = pd.concat([energy,energy_entropy,mfcc_feat,chroma_vector,chroma_deviation],axis=1)
    all = all.as_matrix()

    #mfcc
    mfcc = mfcc_feat.as_matrix()

    #default
    feat = all

    if input_feature == 'MFCC':
        feat = mfcc
    elif input_feature == 'MFCC_CHROMA':
        feat = mfcc_chroma
    elif input_feature == 'ALL':
        feat = all

    model = torch.load(model_file)
    output = model(prepare_sequence(feat))
    output_file = file(feature_file.replace('.csv','va_prediction.csv'),'wb')
    output_idf = csv.writer(output_file)
    if plot:
        plt.figure()
        plt.subplot(1,2,1)
        plt.ylim((0, 1))
        plt.xlabel("time")
        if torch.cuda.is_available():
            plt.plot(output.data[0].cpu().numpy()[:,0],label='valence',color='coral')
        else:
            plt.plot(output.data[0].numpy()[:, 0],label='valence',color='coral')
        plt.subplot(1,2,2)
        plt.ylim((0, 1))
        plt.xlabel("time")
        plt.ylabel("Arousal values")
        if torch.cuda.is_available():
            plt.plot(output.data[0].cpu().numpy()[:,1],label='arousal')
        else:
            plt.plot(output.data[0].numpy()[:, 1],label='arousal')

        plt.suptitle('[' + feature_file + ']' + model_file)
        plt.show()


def prepare_sequence(seq):
    tensor = torch.from_numpy(seq.astype(float)).float()
    return autograd.Variable(tensor).cuda()

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:],'hf:o:l:s:e:i:d:',['file=','optimizer=','loss_fn=','training_size=','epoch=','input=','hidden_size=','help'])

    # default model
    optimizer = 'Adam'
    loss_fn = 'L1'
    training_size = 100
    epoch = 500
    filename = ''
    input_feature = 'MFCC'
    hidden_size = 128

    for key,value in opts:
        if key in ['-h','--help']:
            print('--Automatic Music Emotion Recognition Based on LSTM-RNN model--')
            print('params:')
            print('-h\thelp')
            print('-f\tFile Name')
            print('-o\toptimizer\tAdam, SGD, RMSprop')
            print('-l\tloss function\tMSE,L1')
            print('-s\ttraining size\t0-150')
            print('-e\tepoch')
            print('-i\tinput\tMFCC, MFCC_CHROMA,ALL')
            print('-d\thidden size')
            exit(0)

        if key in ['-f','--file']:
            filename = value
        if key in ['-o', '--optimizer']:
            optimizer = value
        if key in ['-l', '--loss_fn']:
            loss_fn = value
        if key in ['-s', '--training_size']:
            training_size = value
        if key in ['-e', '--epoch']:
            epoch = value
        if key in ['-i','--input']:
            input_feature = value
        if key in ['-d', '--hidden_size']:
            hidden_size = value

    print('optim = ', optimizer, 'loss_fn = ', loss_fn, 'training_size = ', training_size, 'epoch = ', epoch, 'filename = ',filename, 'input_features = ', input_feature,'hidden_size = ', hidden_size)
    model_filename =  str(optimizer) + '_' + str(loss_fn) + '_' + str(training_size) + '_' + str(epoch) + '_' + str(input_feature) + '_' + str(hidden_size) + '.pkt'
    print(model_filename)
    features = feature_extraction(filename)
    neuralMER(features,model_filename,input_feature,True)