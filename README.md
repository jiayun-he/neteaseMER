#neteaseMER

A project for Undergraduate Thesis of Jinan University.

用于暨南大学本科毕业设计的项目

【基于LSTM-RNN 的音乐情感识别】

- Tensorflow
- spider163
- pyAudioAnalysis
- gensim
- jieba

##已完成的功能
使用spider163获取歌单信息 并将评论歌词等需要用到contextual-based （基于文档的）MER的信息存放到数据库中

------2018_03_28更新------
自动将下载的歌曲根据网易云音乐的song_id重命名并获取midterm features （csv和npy文件）

------2018_03_29更新------
添加自动获取歌词及评论关键词并保存到csv中

------2018_03_30更新------
整理文件
comment.py 预处理评论
1. 从csv文件读取评论
2. 使用jieba分词处理评论，并将分词后的评论使用gensim的word2vec生成词向量
3. 将词向量保存为一个完整的参考字典，以及按歌曲分别的单独字典文件
4. 保存使用word2vec生成的模型，便于以后研究使用
5. 添加rnn的小demo

------2018_03_31更新------
整理文件
新增数据模型
model.py
将0330的功能封装到create_model方法中
新增关键词查找功能（给定一个词向量在词库中给出相应的词汇）

------2018_04_05更新------
model.py 大改

get_songinfo_list(): 返回一个list
获取所有song_id?

create_model()方法参数说明:
word_vec_dim: 关键词向量的维度 整数
save_model: 是否保存关键词word2vec模型 bool
save_comment_vec: 是否保存关键词向量模型到csv文件 bool
save_comment_key: 是否保存关键词到csv文件（关键词，词频） bool
save_comment_va: 是否保存关键词的Valence和Arousal值到csv文件中 bool

生成关键词的valence（积极/消极）的值的模型位置：
snownlp/sentiments/sentiments.marshal

arousal（安静/激动）：
arousal.marshal

要获取arousal的值，要先修改snownlp/sentiments/__init__.py中data的路径

concat_all(): 返回pandas的dataframe对象
将VA模型的值和关键词对应放到同一个dataframe里，如果有文件缺失会报错

首次使用时要先create_model （生成对应的csv文件）

目录：
song_acoustic_features_dir = './data/song_acoustic_features/' 音乐特征
songinfo_file_dir = './data/songinfo.csv' 歌曲信息列表
comment_cut_dir = './data/tmp/comment_cut/' 经过jieba分词处理之后的评论
comment_key_dir = './data/key_comment/' 经过jieba通过tf*idf方式获取的评论关键词
comment_vec_dir = './model/key_comment_vec/' 经过word2vec处理获取的关键词向量
comment_model_dir = './model/key_comment_model/' word2vec关键词向量模型
comment_va_dir = './model/key_comment_va/' 通过snownlp获取的V-A模型的值

