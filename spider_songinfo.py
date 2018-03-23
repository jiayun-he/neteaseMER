import pymysql
import os


os.system('spider163 resetdb') #清空数据库
os.system('spider163 get --playlist 111221399') #获取最高人气的粤语歌单的歌曲清单

conn = pymysql.connect(host='localhost',
                             user='root',
                             passwd='LUKEHE051308',
                             db='spider163',
                             charset='utf8',
                             )

cursor = conn.cursor()
sql = "select song_id from music163"
cursor.execute(sql)
songlist = cursor.fetchall()

comment_list = []

for song in songlist:
    #获取歌曲信息（包括歌词）
    os.system('spider163 get -s ' + str(song[0]))
    #获取歌曲热门评论信息
    os.system('spider163 comment -s ' + str(song[0]))

conn.close()