#   coding=utf-8
from pydub import AudioSegment
import os, sys,csv, pymysql,shutil
reload(sys)
sys.setdefaultencoding('utf-8')

conn = pymysql.connect(host='localhost',
                             user='your-username',
                             passwd='your-password',
                             db='spider163',
                             charset='utf8',
                             )

cursor = conn.cursor()
song_dir = "./audiofiles/wav/"

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
    sql = "delete from songinfo where lyrics = '' or comment_all = ''"
    cursor.execute(sql)
    conn.commit()

#delete rows where an actual mp3 song failed downloading (db has records of non-exist files)
def delete_row_with_non_exist_file(song_dir):
    songs = os.listdir(song_dir)
    songinfos = get_songinfo()

    #traverse two lists and find match
    for songinfo in songinfos:
        not_found = 1
        for song in songs:
            if song.find(songinfo[3]) != -1:
                not_found = 0

        if(not_found == 1):
            sql = "delete from songinfo where song_id = " + str(songinfo[1])
            cursor.execute(sql)
            conn.commit()

#delete songs which is not in the dblist (file exists but not in db)
def delete_song_not_presented_in_db(song_dir):
    songs = os.listdir(song_dir)
    for song in songs:
        song_name = str(os.path.splitext(song)[0])
        if (song_name.isdigit()):
           pass
        else:
            shutil.copyfile(song_dir + song, song_dir + "shattered_info/" + song_name + ".wav")
            os.remove(song_dir + song_name + ".wav")
            print(song_name)

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
def rename_songs(song_dir):
    songs = os.listdir(song_dir)
    songinfos = get_songinfo()
    #search songinfo for song id
    for song in songs:
        for songinfo in songinfos:
            if song.find(songinfo[3] + "-" + songinfo[4]) != -1:
                #rename to song id
                old = song_dir + song
                new = song_dir + str(songinfo[1]) + ".wav"
                os.renames(old,new)

#fill the songinfo table in database (merged all comments to one string)
def create_songinfo():
    sql = "select song_id, song_name, author from music163"
    cursor.execute(sql)
    songlist = cursor.fetchall()
    songinfolist = []
    conn.commit()
    for index, song in enumerate(songlist):
        song_id = song[0]
        song_name = song[1]
        author = song[2]
        songinfolist.append([])
        comment_merged = ""
        # print(index,song)
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
        cursor.execute(sql, (song_id, lyrics, song_name, author, comment_merged))
        conn.commit()

#store songinfo data to csv file
def saveToCsv():
    songinfos = get_songinfo()
    csvfile = file('data/songinfo.csv','wb')
    writers = csv.writer(csvfile)
    writers.writerow(['id','song_id','lyrics','song_name','author','comment_all'])
    if len(songinfos) == 0:
        print("There is no record in table songinfo!")
    for songinfo in songinfos:
        writers.writerow([songinfo[0],songinfo[1],songinfo[2],songinfo[3],songinfo[4],songinfo[5]])
    csvfile.close()

create_songinfo()
delete_missing()
#delete_row_with_non_exist_file(song_dir)
#rename_songs(song_dir)
#delete_song_not_presented_in_db(song_dir)
saveToCsv()