#encoding:utf-8
import pandas as pd
import jieba.analyse
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import sys,csv
reload(sys)
sys.setdefaultencoding('utf-8')
songinfo_df = pd.read_csv('./data/songinfo.csv')
#id, song_id, lyrics, song_name, author, comment_all

def comment():
    #show corrspondence between vector to words
    word_model_csv = file('model/word_vec_dict.csv','wb')
    word_model_writer = csv.writer(word_model_csv)
    for index, row in songinfo_df.iterrows():
        try:
            #read the corrsponding song feature file
            song_dir = './data/songs/csv/' + str(row[1]) + '.wav.csv'
            song_df = pd.read_csv(song_dir)
            total_cols = song_df.shape[0]
        except IOError:
            print("File" + song_dir +' does not exist!')
            continue

        print("------------------------------")
        print("processing comments of song: " + str(row[3]) + "-" +  str(row[4]))
        comments = filter(lambda x: x not in '0123456789:.', str(row[5]))

        #store the preprocessed comments for later modeling
        comments_cut = jieba.cut(comments)
        result = ' '.join(comments_cut)
        result = result.encode('utf-8')
        with open('./data/comment_cut/' + str(row[1]) + '.txt', 'w') as f2:
            f2.write(result)

        sentences = LineSentence('./data/comment_cut/' + str(row[1]) + '.txt')
        model = Word2Vec(sentences, hs=1, min_count=1, window=3, size=10)

        # extract keyword with tf-idf method
        key_words = jieba.analyse.extract_tags(comments,topK=total_cols,withWeight=True)
        csvfile_idf = file('data/keywords/comments/tf_idf/' + str(row[1]) + '.csv','wb')
        writers_idf = csv.writer(csvfile_idf)
        #store dict info as separate files
        csvfile_model = file('model/comments/' + str(row[1]) + '.csv','wb')
        writers_model = csv.writer(csvfile_model)
        print("number of keywords: " + str(len(key_words)))
        for key_word in key_words:
            writers_idf.writerow(key_word)
            print("vector for word '" +  key_word[0] + "'")
            writers_model.writerow([key_word[0],model[key_word[0]]])
            word_model_writer.writerow([key_word[0],model[key_word[0]]])
            model.save('model/' + str(row[1]) + '.model')

        del model

def get_key_lyrics():
    for index, row in songinfo_df.iterrows():
        try:
            #read the corrsponding song feature file
            song_dir = './data/songs/csv/' + str(row[1]) + '.wav.csv'
            song_df = pd.read_csv(song_dir)
            total_cols = song_df.shape[0]
        except IOError:
            print("File" + song_dir +' does not exist!')
            continue
        #extract with tf-idf method
        #delete all numbers and symbols
        lyrics = filter(lambda x:x not in '0123456789:.',str(row[2]))
        key_words = jieba.analyse.extract_tags(lyrics,topK=total_cols+41,withWeight=True)
        csvfile_idf = file('data/keywords/lyrics/tf_idf/' + str(row[1]) + '.csv','wb')
        writers_idf = csv.writer(csvfile_idf)
        writers_idf.writerow([row[3] + "-" + row[4]])
        writers_idf.writerow(['word','frequency'])
        for key_word in key_words:
            writers_idf.writerow(key_word)

comment()

