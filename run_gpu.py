# coding=utf-8
from pyAudioAnalysis import audioFeatureExtraction
import sys,getopt,os
import torch,csv
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
reload(sys)
sys.setdefaultencoding('utf-8')


def feature_extraction(filename):
    wavFileName = filename.replace('.mp3','.wav')
    name = filename.replace('.mp3','')
    command = 'avconv -i ' + filename + ' ' + wavFileName
    os.system(command)
    audioFeatureExtraction.mtFeatureExtractionToFile(wavFileName,2.0, 2.0, 0.050, 0.050,name,storeStFeatures=True,storeToCSV=True)
    return name + '.csv'

def neuralMER(feature_file,model_file,input_feature,plot = True):
    features = pd.read_csv(feature_file)
    mfcc_feat = features.iloc[:,8:21]
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

    if input_feature == 'MFCC':
        feat = mfcc
    elif input_feature == 'MFCC_CHROMA':
        feat = mfcc_chroma
    elif input_feature == 'ALL':
        feat = all

    model = torch.load(model_file)
    print(model)
    output = model(prepare_sequence(feat))
    output_file = file(feature_file.replace('.csv','_va_prediction.csv'),'wb')
    output_idf = csv.writer(output_file)
    valence_values = output.data[0].cpu().numpy()[:,0]
    arousal_values = output.data[0].cpu().numpy()[:,1]
    output_idf.writerow(['time','valence_value','arousal_value'])
    for i in range(valence_values.size):
        output_idf.writerow([i,valence_values[i],arousal_values[i]])

    if plot:
        plt.figure(figsize=(40,5))
        plt.suptitle('[' + unicode(feature_file) + ']' + model_file)
        plt.subplot(1,2,1)
        plt.xlabel("time")
        plt.ylabel("Valence values")
        if torch.cuda.is_available():
            plt.plot(valence_values,label='valence',color='coral')
        else:
            plt.plot(arousal_values,label='valence',color='coral')
        plt.subplot(1,2,2)
        plt.xlabel("time")
        plt.ylabel("Arousal values")
        if torch.cuda.is_available():
            plt.plot(arousal_values,label='arousal')
        else:
            plt.plot(arousal_values,label='arousal')

        plt.show()
    print('peak:',getPeak(valence_values,arousal_values,0.1))
    print('valley:',getValley(valence_values, arousal_values, 0.1))
    print('turning density:',getTurningDensity(valence_values,arousal_values,0.05))
    print('range:',getRange(valence_values,arousal_values))


def prepare_sequence(seq):
    tensor = torch.from_numpy(seq.astype(float)).float()
    return autograd.Variable(tensor).cuda()

def getRange(valence_arr,arousal_arr):
    return np.max(valence_arr) - np.min(valence_arr), np.max(arousal_arr) - np.min(arousal_arr)
#Input array is a 2-dimensional array
def getPeak(valence_arr,arousal_arr,metric):
    valence_max = np.max(valence_arr)
    valence_accumulate = 0
    arousal_max = np.max(arousal_arr)
    arousal_accumulate = 0
    length = valence_arr.shape[0]
    for v in valence_arr:
        if valence_max - v <= metric:
            valence_accumulate += v
    for a in arousal_arr:
        if arousal_max - a <= metric:
            arousal_accumulate += a

    return valence_accumulate/length, arousal_accumulate/length

def getValley(valence_arr,arousal_arr,metric):
    valence_min = np.min(valence_arr)
    valence_accumulate = 0
    arousal_min = np.min(arousal_arr)
    arousal_accumulate = 0
    length = valence_arr.shape[0]
    for v in valence_arr:
        if v - valence_min <= metric:
            valence_accumulate += v
    for a in arousal_arr:
        if arousal_min - a <= metric:
            arousal_accumulate += a

    return valence_accumulate/length, arousal_accumulate/length

def getTurningDensity(valence_arr,arousal_arr,metric):
    v_count = 0.0
    a_count = 0.0
    length = valence_arr.shape[0]
    for index, value in enumerate(valence_arr):
        if index+1 < valence_arr.shape[0]:
            if (valence_arr[index+1] - valence_arr[index]) > metric:
                v_count += 1
            if (arousal_arr[index+1] - arousal_arr[index]) > metric:
                a_count += 1

    return v_count/length, a_count/length


def test():
    # default model
    optimizer = 'SGD'
    loss_fn = 'MSE'
    epoch = 50
    filename = ''
    input_feature = 'MFCC'
    filename = '一事无成.mp3'

    print(
    'optim = ', optimizer, 'loss_fn = ', loss_fn, 'epoch = ', epoch, 'filename = ',
    filename, 'input_features = ', input_feature)
    model_filename = 'pretrained/' + str(optimizer) + '_' + str(loss_fn)  + '_' + str(
        epoch) + '_' + str(input_feature) + '.pkt'
    print(model_filename)
    features = feature_extraction(filename)
    neuralMER(features, model_filename, input_feature, False)

test()

# if __name__ == '__main__':
#     opts, args = getopt.getopt(sys.argv[1:],'hf:o:l:s:e:i:d:',['file=','optimizer=','loss_fn=','epoch=','input=','help'])
#
#     # default model
#     optimizer = 'Adam'
#     loss_fn = 'L1'
#     epoch = 500
#     filename = ''
#     input_feature = 'MFCC'
#
#     for key,value in opts:
#         if key in ['-h','--help']:
#             print('--Automatic Music Emotion Recognition Based on LSTM-RNN model--')
#             print('params:')
#             print('-h\thelp')
#             print('-f\tFile Name')
#             print('-o\toptimizer\tAdam, SGD')
#             print('-l\tloss function\tMSE,L1')
#             print('-e\tepoch')
#             print('-i\tinput\tMFCC, MFCC_CHROMA,ALL')
#             exit(0)
#
#         if key in ['-f','--file']:
#             filename = value
#         if key in ['-o', '--optimizer']:
#             optimizer = value
#         if key in ['-l', '--loss_fn']:
#             loss_fn = value
#         if key in ['-e', '--epoch']:
#             epoch = value
#         if key in ['-i','--input']:
#             input_feature = value
#
#     print('optim = ', optimizer, 'loss_fn = ', loss_fn, 'epoch = ', epoch, 'filename = ',filename, 'input_features = ', input_feature)
#     model_filename =  'pretrained/' + str(optimizer) + '_' + str(loss_fn) + '_' + str(epoch) + '_' + str(input_feature) + '.pkt'
#     print(model_filename)
#     features = feature_extraction(filename)
#     neuralMER(features,model_filename,input_feature,True)