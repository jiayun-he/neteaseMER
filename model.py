import numpy as np
import jieba.analyse
import pandas as pd
import pickle
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from snownlp import SnowNLP
import sys,csv,os
reload(sys)
sys.setdefaultencoding('utf-8')

song_acoustic_features_dir = './data/song_acoustic_features/'
songinfo_file_dir = './data/songinfo.csv'
comment_cut_dir = './data/tmp/comment_cut/'
comment_key_dir = './data/key_comment/'
comment_vec_dir = './model/key_comment_vec/'
comment_model_dir = './model/key_comment_model/'
comment_va_dir = './model/key_comment_va/'

class DataSet:
    def __init__(self,word_vec_dim = 1, create_models = False, save_model = False, save_comment_key = False, save_comment_vec = False, save_comment_va = False):
        self._index_in_epoch = 0
        if create_models:
            self.create_model(word_vec_dim, save_model,save_comment_key, save_comment_vec, save_comment_va)

    def get_songid_list(self):
        songid_list = []
        valence_files = os.listdir(comment_va_dir + 'valence/')
        for valence_file in valence_files:
            songid_list.append(filter(str.isdigit,valence_file))

        return songid_list

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._data[start:end]

    #convert number to corresponding word
    def vec2word(self,word_vec):
        print(word_vec)

        for key_comment_vec in self._key_comment_vecs:
        # songid, keyword(string), vector
            if word_vec.all() == key_comment_vec[2].all():
                return key_comment_vec[1]

        return 'Corresponding keyword Not Found!'

    #convert comment to vector model
    def create_model(self,word_vec_dim = 1, save_model = False, save_comment_vec = False, save_comment_key = False, save_comment_va = False):
        songinfo_df = pd.read_csv(songinfo_file_dir)
        comment_vec_dict_csv = file(comment_vec_dir + 'dict.csv','wb')
        comment_vec_dict_writer = csv.writer(comment_vec_dict_csv)
        comment_vec_dict_writer.writerow(['keyword','vector'])

        for index, row in songinfo_df.iterrows():
            # id, song_id, lyrics, song_name, author, comment_all
            try:
                # read the corrsponding song feature file
                song_dir = song_acoustic_features_dir + str(row[1]) + '.wav.csv'
                song_df = pd.read_csv(song_dir)
                total_cols = song_df.shape[0]
            except IOError:
                #print("File" + song_dir + ' does not exist!')
                continue

            print("------------------------------")
            print("processing comments of song: " + str(row[3]) + "-" + str(row[4]))
            comments = filter(lambda x: x not in '0123456789:.', str(row[5]))

            # store the preprocessed comments for later modeling
            comments_cut = jieba.cut(comments)
            result = ' '.join(comments_cut)
            result = result.encode('utf-8')
            with open(comment_cut_dir + str(row[1]) + '.txt', 'w') as f2:
                f2.write(result)

            sentences = LineSentence(comment_cut_dir + str(row[1]) + '.txt')
            model = Word2Vec(sentences, hs=1, min_count=1, window=3, size=word_vec_dim)

            # extract keyword with tf-idf method
            key_word_single_file = jieba.analyse.extract_tags(comments, topK=total_cols)
            #print("number of keywords: " + str(len(key_word_single_file)))
            if save_comment_key:
                csvfile_idf = file(comment_key_dir + str(row[1]) + '.csv', 'wb')
                writers_idf = csv.writer(csvfile_idf)
                writers_idf.writerow(['keyword','freq'])
            # store dict info as separate files
            if save_comment_vec:
                csvfile_model = file(comment_vec_dir + str(row[1]) + '.csv', 'wb')
                writers_model = csv.writer(csvfile_model)
                writers_model.writerow(['keyword','vector'])

            if save_comment_va:
                csvfile_valence = file(comment_va_dir + 'valence/' + str(row[1]) + '.csv', 'wb')
                writer_va = csv.writer(csvfile_valence)
                writer_va.writerow(['keyword','valence_value'])
                #csvfile_arousal = file(comment_va_dir + str(row[1]) + '_valence.csv', 'wb')
            for key_word in key_word_single_file:
                s = SnowNLP(key_word)
                arousal_value = s.sentiments
                if save_comment_vec:
                    writers_idf.writerow(key_word)
                #print("vector for word '" + key_word[0] + "'")
                if save_comment_vec:
                    writers_model.writerow([key_word, model[key_word]])

                if save_comment_va:
                    writer_va.writerow([key_word,arousal_value])

                comment_vec_dict_writer.writerow([key_word, model[key_word]]) #create word vec dictionary

                if save_model:
                    model.save(comment_model_dir + str(row[1]) + '.model')

            del model

        return True

    #
    def concat_all(self, song_id):
        valence = self.load_valence(song_id)
        arousal = self.load_arousal(song_id)
        key_comment_vector = self.load_key_comment_vec(song_id)
        return pd.concat([valence,arousal['arousal_value'],key_comment_vector['vector']],axis=1,join='inner')

    #return a pandas dataframe given a song_id
    def load_valence(self, song_id):
        valence_file = comment_va_dir + 'valence/'+ song_id + '.csv'
        try:
            df = pd.read_csv(valence_file)
        except IOError:
            print(song_id + ' valence model not found!')
            exit(1)
        else:
            return df

    def load_arousal(self,song_id):
        arousal_file = comment_va_dir + 'arousal/' + song_id + '.csv'
        try:
            df = pd.read_csv(arousal_file)
        except IOError:
            print(song_id + ' arousal model not found!')
            exit(1)
        else:
            return df

    def load_song_features(self,song_id):
        song_feat_file = song_acoustic_features_dir + song_id + '.wav.csv'
        try:
            df = pd.read_csv(song_feat_file)
        except IOError:
            print(song_id + ' song_feat file not found!')
            exit(1)
        else:
            return df

    def load_key_comment_vec(self,song_id):
        key_comment_vec_file = comment_vec_dir + song_id + '.csv'
        try:
            df = pd.read_csv(key_comment_vec_file)
        except IOError:
            print(song_id + ' comment key vector file not found!')
            exit(1)
        else:
            return df


