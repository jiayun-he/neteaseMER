import tensorflow as tf
import sys,csv
import pandas as pd
import numpy as np
from model import DataSet
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
for song_id in song_ids:
    labels = dataset.concat_all(song_id)
    song_feats = dataset.load_song_features(song_id)
    num_of_rows = min(labels.shape[0],song_feats.shape[0])
    mfcc_data = song_feats.iloc[0:num_of_rows,10:22]



#word_vec = np.array([0.35129613],dtype=np.float32)
#print(dataset.vec2word(word_vec))