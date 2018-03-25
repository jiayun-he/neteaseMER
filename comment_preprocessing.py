import pymysql,sys
reload(sys)
sys.setdefaultencoding('utf-8')

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
for index,song in enumerate(songlist):
    song_id = song[0]
    song_name = song[1]
    author = song[2]
    songinfolist.append([])
    comment_merged = ""
    #print(index,song)
    sql = "select txt from comment163 where song_id = " + str(song_id)
    cursor.execute(sql)
    comment_list = cursor.fetchall()
    for comment in comment_list:
        comment_merged += comment[0]

    conn.commit()
    sql = "select txt from lyric163 where song_id = " + str(song_id)
    cursor.execute(sql)
    lyrics = cursor.fetchone()

    sql = "insert into songinfo (song_id,lyrics,song_name,author,comment_all) VALUES (%s,%s,%s,%s,%s)"
    cursor.execute(sql,(song_id,lyrics,song_name,author,comment_merged))
    conn.commit()

conn.close()