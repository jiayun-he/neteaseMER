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
comment_model_csv_dir = './model/key_comment_csv/'
comment_model_dir = './model/key_comment_model/'
comment_va_dir = './model/key_comment_va/'

#tmp model files
_models_dir = './data/tmp/_models.obj'
_key_dir = './data/tmp/_key_comment_vecs.obj'
_valence_dir = './data/tmp/_valence.obj'
_arousal_dir = './data/tmp/_arousal.obj'
_labels_dir = './data/tmp/_labels.obj'
class DataSet:
    def __init__(self,word_vec_dim = 1, create_models = False, save_model = False, save_model_csv = False, save_comment_key_csv = False, save_comment_va = False):
        self._index_in_epoch = 0
        if create_models:
            self._models,self._key_comment_vecs = self.create_model(word_vec_dim, save_model,save_model_csv, save_comment_key_csv, save_comment_va)
            self._valence, self._arousal = self.load_va_model()
            key_filehandler = open(_key_dir,'w')
            pickle.dump(self._key_comment_vecs,key_filehandler)
            model_filehandler = open(_models_dir,'w')
            pickle.dump(self._models,model_filehandler)
            valence_filehandler = open(_valence_dir,'w')
            pickle.dump(self._valence,valence_filehandler)
            arousal_filehandler = open(_arousal_dir,'w')
            pickle.dump(self._arousal,arousal_filehandler)
            #label_filehandler = open(_labels_dir,'w')
            #pickle.dump(self._output,label_filehandler)

        #label_filehandler = open(_labels_dir)
        #self._output = pickle.load(label_filehandler)
        key_filehandler = open(_key_dir, 'r')
        self._key_comment_vecs = pickle.load(key_filehandler)
        model_filehandler = open(_models_dir, 'r')
        self._models = pickle.load(model_filehandler)
        valence_filehandler = open(_valence_dir, 'r')
        self._valence = pickle.load(valence_filehandler)
        arousal_filehandler = open(_arousal_dir, 'r')
        self._arousal = pickle.load(arousal_filehandler)
        self._songid_list = self.get_songid_list()
        self._output = self.arrange_data()

    def arrange_data(self):
        output = []
        for songid in self._songid_list:
            for index in range(len(self._valence)):
                if self._valence[index][0] == songid:
                    #one_song_key_vec = self._key_comment_vecs[index][2]
                    one_song_keyword = self._valence[index][1]
                    one_song_valence = self._valence[index][2]
                    one_song_arousal = self._arousal[index][2]
                    output.append([songid, one_song_keyword,one_song_valence, one_song_arousal])

        return output


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
    def create_model(self,word_vec_dim = 1, save_model = False, save_model_csv = False, save_comment_key_csv = False, save_comment_va = False):
        songinfo_df = pd.read_csv(songinfo_file_dir)
        comment_model_csv = file(comment_model_csv_dir + 'dict.csv','wb')
        comment_model_writer = csv.writer(comment_model_csv)
        models = []
        keyword_word_vecs = []
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
            models.append([row[1],model]) #songid, word2vec_model

            # extract keyword with tf-idf method
            key_word_single_file = jieba.analyse.extract_tags(comments, topK=total_cols)
            #print("number of keywords: " + str(len(key_word_single_file)))
            if save_comment_key_csv:
                csvfile_idf = file(comment_key_dir + str(row[1]) + '.csv', 'wb')
                writers_idf = csv.writer(csvfile_idf)
            # store dict info as separate files
            if save_model_csv:
                csvfile_model = file(comment_model_csv_dir + str(row[1]) + '.csv', 'wb')
                writers_model = csv.writer(csvfile_model)

            if save_comment_va:
                csvfile_arousal = file(comment_va_dir + str(row[1]) + '_valence.csv', 'wb')
                writer_va = csv.writer(csvfile_arousal)
                #csvfile_arousal = file(comment_va_dir + str(row[1]) + '_valence.csv', 'wb')
            for key_word in key_word_single_file:
                s = SnowNLP(key_word)
                arousal_value = s.sentiments
                if save_comment_key_csv:
                    writers_idf.writerow(key_word)
                #print("vector for word '" + key_word[0] + "'")
                if save_model_csv:
                    writers_model.writerow([key_word, model[key_word]])

                if save_comment_va:
                    writer_va.writerow([key_word,arousal_value])
                comment_model_writer.writerow([key_word, model[key_word]]) #create word vec dictionary
                keyword_word_vecs.append([row[1],key_word, model[key_word]]) #songid, keyword(string), vector

                if save_model:
                    model.save(comment_model_dir + str(row[1]) + '.model')

            del model

        return models,keyword_word_vecs

    def load_va_model(self):
        valence_files = os.listdir(comment_va_dir + 'valence/')
        arousal_files = os.listdir(comment_va_dir + 'arousal/')
        valence_model = []
        arousal_model = []

        for valence_file in valence_files:
            song_id = filter(str.isdigit,valence_file)
            valence_df = pd.read_csv(comment_va_dir + 'valence/' + valence_file)
            for index, row in valence_df.iterrows():
                valence_model.append([song_id,row[0],row[1]]) #songid, keyword, valence_value

        for arousal_file in arousal_files:
            song_id = filter(str.isdigit,arousal_file)
            arousal_df = pd.read_csv(comment_va_dir + 'arousal/' + arousal_file)
            for index, row in arousal_df.iterrows():
                arousal_model.append([song_id,row[0],row[1]]) #songid, keyword, arousal_value

        return valence_model, arousal_model

    def get_labels(self):
        return self._output


