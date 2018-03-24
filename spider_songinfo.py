import pymysql
import os

#os.system('spider163 resetdb') #delete current db
#os.system('spider163 get --playlist 111221399')

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
    #obtain song information
    os.system('spider163 get -s ' + str(song[0]))
    #obtain song comments
    os.system('spider163 comment -s ' + str(song[0]))

conn.close()