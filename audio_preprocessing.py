#   coding=utf-8
from pydub import AudioSegment
import os, sys,csv, pymysql
reload(sys)
sys.setdefaultencoding('utf-8')

conn = pymysql.connect(host='localhost',
                             user='root',
                             passwd='LUKEHE051308',
                             db='spider163',
                             charset='utf8',
                             )

cursor = conn.cursor()

def create_songlist():
    sql = "select song_id, author, song_name from music163"
    cursor.execute(sql)
    songlist = cursor.fetchall()
    c = csv.writer(open("songlist.csv","wb"))
    for index in range(len(songlist)):
        c.writerow(songlist[index])

#retrieve song data from database
def get_songinfo():
    sql = "select * from songinfo"
    cursor.execute(sql)
    return cursor.fetchall()

#delete incomplete data from database
def delete_missing():
    sql = "delete from songinfo where isnull(lyrics) or isnull(comment_all)"
    cursor.execute(sql)
    conn.commit()

#cut all mp3 pieces to 3 segments
def triple_cut(song_dir,segment_dir):
    songs = os.listdir(song_dir)
    for song in songs:
        audiofilepath = song_dir + song
        music = AudioSegment.from_mp3(audiofilepath)
        print("Length: ", music.duration_seconds, "seconds")
        #cut each music piece to 3 segments and store them into another folder
        segment_length = (music.duration_seconds / 3) * 1000
        music[0:segment_length].export(segment_dir + file.replace(".mp3","_1.mp3"), format="mp3")
        music[segment_length:segment_length * 2].export(segment_dir + file.replace(".mp3","_2.mp3"), format="mp3")
        music[segment_length * 2:].export(segment_dir + file.replace(".mp3","_3.mp3"), format="mp3")

# rename all audio files to song_id
def rename(song_dir):
    songs = os.listdir(song_dir)
    songinfos = get_songinfo()
    for songinfo in songinfos:
        for song in songs:
            if str.find(song, songinfo[3]):
                os.rename(song_dir + song, str(songinfo[1]) + ".wav")





