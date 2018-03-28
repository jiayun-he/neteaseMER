#encoding:utf-8
import pandas as pd
import numpy as np
import jieba.analyse
import sys,csv
reload(sys)
sys.setdefaultencoding('utf-8')
songinfo_df = pd.read_csv('./data/songinfo.csv')
#id, song_id, lyrics, song_name, author, comment_all

def get_key_comment():
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
        comments = filter(lambda x: x not in '0123456789:.', str(row[5]))
        key_words = jieba.analyse.extract_tags(comments,topK=total_cols,withWeight=True)
        csvfile_idf = file('data/keywords/comments/tf_idf/' + str(row[1]) + '.csv','wb')
        writers_idf = csv.writer(csvfile_idf)
        writers_idf.writerow([row[3] + "-" + row[4]])
        writers_idf.writerow(['word','frequency'])
        for key_word in key_words:
            writers_idf.writerow(key_word)

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
        key_words = jieba.analyse.extract_tags(lyrics,topK=total_cols+40,withWeight=True)
        csvfile_idf = file('data/keywords/lyrics/tf_idf/' + str(row[1]) + '.csv','wb')
        writers_idf = csv.writer(csvfile_idf)
        writers_idf.writerow([row[3] + "-" + row[4]])
        writers_idf.writerow(['word','frequency'])
        for key_word in key_words:
            writers_idf.writerow(key_word)

get_key_comment()
get_key_lyrics()
