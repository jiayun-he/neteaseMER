# coding=utf-8
from pyAudioAnalysis import audioFeatureExtraction
import os,sys
reload(sys)
sys.setdefaultencoding('utf-8')

mp3_songdir = "/audiofiles/mp3/"
wav_songdir = "./audiofiles/wav/"
#modified version of convert mp3 files to wav files
def myConvertDirMP3ToWav(dirName, Fs, nC, useMp3TagsAsName = False):
    filenames = os.listdir(dirName)
    for f in filenames:
        dir = dirName + f
        wavFileName = dir.replace(".mp3",".wav")
        command = "avconv -i \"" + dir + "\" -ar " +str(Fs) + " -ac " + str(nC) + " \"" + wavFileName + "\"";
        print command
        os.system(command)

#myConvertDirMP3ToWav(mp3_songdir,16000,2)#(directory,sampling rate,number_of_channels,use_mp3_tag_name_as_filename)
#command = "mv " + mp3_songdir + "*.wav " + wav_songdir
#os.system(command)
#audioFeatureExtraction.mtFeatureExtractionToFileDir(wav_songdir, 2.0, 2.0, 0.050, 0.050, True, True)
#os.system("mv " + wav_songdir + "*.csv data/csv")
#os.system("mv " + wav_songdir + "*.npy data/npy")
