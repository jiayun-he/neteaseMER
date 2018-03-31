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