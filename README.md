### Short Video-Analytics

#### Introduction
   ###### 1. This small project is conducted to analyze the candidate's answer during video interview.
   ###### 2. Project used existing pre-trained models 
            * For video/image emotion analysis : FER library is used which has pre-trained model based on Harcascade and MTCNN
            * For speech to Text analysis :Speech_recognition library is used
            * For text sentiment analysis :VADER from ntlk library is used.VADER(Valence Aware Dictionary and sentiment Reasoner)
   ###### 3. The Analysis is presented in the form of Heat-Maps showing neagtive-positive sentiments from words and negative and positive emotions of face in video.

#### Emotions-Sentiment Heat Maps

###### The sample of result/outcome of the code.

![](https://github.com/RammySekham/Short-Video-Analytics/blob/main/Images/HeatMaps.PNG)

#### How to run this code
       1. Select directory ($ cd <directory>)
       2. Clone the repo  ($ git clone <repo-url>)
       3. Install dependencies  
          pip install -r requirements.txt 
                OR individual modules using 
          pip install <module_name>
       4. Place mp4 video file of small size 20-40sec introductory video in your working directory
       5. Change file_name in settings.py file
       6. Run the code "Video_Analytics.py"
