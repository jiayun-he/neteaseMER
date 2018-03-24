#整合歌曲信息并存入数据库
import pymysql
conn = pymysql.connect(host='localhost',
                             user='root',
                             passwd='LUKEHE051308',
                             db='spider163',
                             charset='utf8',
                             )

cursor = conn.cursor()

sql = "select song_id, song_name, author from music163"
cursor.execute(sql)
songlist = cursor.fetchall()
songinfolist = []
conn.commit()
#遍历清单中的所有歌曲
for index,song in enumerate(songlist):
    song_id = song[0]
    song_name = song[1]
    author = song[2]
    songinfolist.append([])
    #初始化该歌曲的评论string
    comment_merged = ""
    #print(index,song)
    #获取该歌曲的评论信息
    sql = "select txt from comment163 where song_id = " + str(song_id)
    cursor.execute(sql)
    #合并同一首歌曲的所有评论
    comment_list = cursor.fetchall()
    for comment in comment_list:
        comment_merged += comment[0]

    conn.commit()
    #获取歌曲的歌词
    sql = "select txt from lyric163 where song_id = " + str(song_id)
    cursor.execute(sql)
    lyrics = cursor.fetchone()

    #汇总到新表中
    sql = "insert into songinfo (song_id,lyrics,song_name,author,comment_all) VALUES (%s,%s,%s,%s,%s)"
    cursor.execute(sql,(song_id,lyrics,song_name,author,comment_merged))
    cursor.commit()

conn.close()