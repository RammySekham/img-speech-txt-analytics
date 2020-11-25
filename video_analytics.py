#!/usr/bin/env python
import settings                              # Importing required modules
import cv2                                   # OpenCV # For face emotion detection
from fer import FER                          # https://pypi.org/project/fer/
import matplotlib.pyplot as plt              # MatPlotLib for plots
from moviepy.editor import VideoFileClip     # Importing from MoviePy and datetime
import datetime
import pandas as pd                          
import numpy as np
import seaborn as sns                        # Seaborn for Heat-Maps
import os                                    # for Speech to text: imports
import subprocess
import speech_recognition as sr
import ffmpeg
import shlex
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_video():
    
    ''' getting video file from directory'''
    
    myvideo = settings.video_file
    return myvideo


def emotions_face_video(myvideo):
    
    ''' Functions to return dictonary of bounding
    boxes for faces, emotions and scores'''
    
    clip = VideoFileClip(myvideo)
    duration = clip.duration                         
    vidcap = cv2.VideoCapture(myvideo)                      # VideCapture from cv2
    i = 0                                                   # initiate the variable for loop, will run for number of frames/images
    d = []                                                  # dictionary to capture the input of each image
    sec = 0                                                 # Variable to capture frame at particular time in the video
    frameRate = 1.0                                         # frameRate, to alter the time at which the frame is captured
    while i < abs((duration/frameRate) + 1):                # Numebr of frames based on duration and frameRate
            sec = sec + frameRate
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)     #Capturing video at particular intervals
            ret, image = vidcap.read() 
            if ret:                                         # If it has a frame
                    cv2.imwrite("image.jpg", image)         # saving image
                    img = plt.imread("image.jpg")           # reading image
                    detector = FER()                        # Calling fer for using already trained model
                    d = d + detector.detect_emotions(img)   # dictionary to store output of each image
            i = i + 1                                       # incrementing Loop
    return d



def emotion_face_video_dataframe(d):
    
    
    ''' Sentiment Analysis based on emotion detection 
      returns list of dictionaries for each image emotions and scores '''
    
     
    m = len(d)                                                                    # Get length of the dictinary
    cols =['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']     # Based on Trained Model in Fer
    df = pd.DataFrame(columns = cols)                                             # initiate empty dataframe
    list_dict = []                                                                # initate empty dictionary
    for i in range(0,m): 
        list_dict.append(d[i]['emotions'])                                  # Appending data from nested dictionary for each image in a list
    df= df.append(list_dict)                                                # Appending data from list into a empty dataframe
    return df



def speech_to_text(myvideo):
    
    '''converts speech to text using recognizer from Google  '''

    command = "ffmpeg -i "  + myvideo +  " Test4.mp3"                       # Command line to convert mp4 file into mp3
    args = shlex.split(command)                                             # Split the args as required by subprocess
    subprocess.run(args)  

    command = "ffmpeg -i Test4.mp3  Test4.wav"                              # Command line to convert to mp3 into wav file
    args = shlex.split(command)
    subprocess.run(args)

    r = sr.Recognizer()                                                     # Making an instance of Recognizer
    with sr.AudioFile('Test4.wav') as source:               
        audio = r.record(source, duration=50)                               # duration 100 secs
        try:
            text_output = r.recognize_google(audio, language='en-IN')
        except Exception as e:
            print("could not understand audio")                             # Exception for empty audio or other lanaguage
    return text_output
    

def sentiment_Analysis_text(text_output):
    
    ''' Sentiment Analysis on the text'''
    
    nltk.download('vader_lexicon')                                   # https://www.kaggle.com/nltkdata/vader-lexicon
                                                                     # VADER Sentiment Analysis. 
                                                                     # VADER (Valence Aware Dictionary and sentiment Reasoner) is a lexicon 
    senti = SentimentIntensityAnalyzer()                             # and rule-based sentiment analysis tool 
    senti_text = senti.polarity_scores(text_output)                  # Give {'neg': 0.083, 'neu': 0.565, 'pos': 0.352, 'compound': 0.8733}
                                                                  
    stopwords = ["a", "this", "is", "and", "i", "to", "for", "very",                 # Excluding stopwords from text output(text to speech)
                 "know", "all", "the", "here", "about", "people", "you", "that"]
    
    reduced = list(filter(lambda w: w not in stopwords, (text_output.lower()).split()))
    
    data =({
    "Words":["Paragraph"] + reduced,
    "Sentiment":[senti_text["compound"]] + [senti.polarity_scores(word)["compound"] 
                                            for word in reduced]
     }) 
    
    return senti_text, data


def video_sentiments(df_faces):
    
    '''Using Seaborn to show heatmap of emotions with their probablities for video '''

    fig, ax = plt.subplots(figsize=(10,10)) 
    sns.heatmap(df_faces, annot=True)
   
    return None

   
def text_sentiments(data):
    
    ''' returns heatmap of text sentiments for text'''
    
    grid_kws = {"height_ratios": (0.1, 0.007), "hspace": 2}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    f.set_figwidth(20)
    f.set_figheight(3)
    sns.heatmap(pd.DataFrame(data).set_index("Words").T,center=0, ax=ax, 
                annot=True, cbar_ax=cbar_ax, cbar_kws={"orientation": "horizontal"}, cmap = "PiYG")
    return None

if __name__ == "__main__":
    
    myvideo = get_video()                                         # Small video 40-60 secs
    d = emotions_face_video(myvideo)                              # Calling emotions_face_video, returns list of emotion- score dictionary 
    df = emotion_face_video_dataframe(d)                          # Dictionary to dataframe
    video_sentiments(df)                                          # Heat Map


    text_output = speech_to_text(myvideo)                         # Converting speech to text
    print(text_output)                                            # Converted text
    senti_text, data = sentiment_Analysis_text(text_output)       #score and data for heatmap
    print(senti_text)                                             # overall positive, negative score
    text_sentiments(data)                                         # Heat Map
