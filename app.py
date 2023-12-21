import streamlit as st
import matplotlib.pyplot as plt  
import seaborn as sn
import pyaudio
import wave
import numpy as np
import pandas as pd
import librosa
from python_speech_features import mfcc ,logfbank
from scipy.io import wavfile
import os
from cfg import Config
from keras.models import load_model
import pickle
from IPython.display import display, Audio
import time
from scipy.io import wavfile



def get_sample_audio():
    chunks=1024
    formats=pyaudio.paInt16
    channel=1
    rate=48000
    audio=pyaudio.PyAudio()

    stream=audio.open(format=formats,channels=channel,rate=rate,input=True,frames_per_buffer=chunks)

    # print('start recording ..')
    frames=[]
    seconds=3
    # st.sidebar.success('Start Recording!')
    print('start recording ..')
    
    for _ in range(0,int(rate / chunks*seconds)):
       
        data=stream.read(chunks)
        frames.append(data)

    # st.sidebar.error('Stop Recording!')
    
    print('stop recording ..')
    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf=wave.open('sample.wav','wb')
    wf.setnchannels(channel)
    wf.setsampwidth(audio.get_sample_size(formats))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def down_sampling():
    signal,rate_x=librosa.load('sample.wav',sr=16000)  # down sampling because thier not much data at higher freq 
    
    mask=envelope(signal,rate_x,0.0005)
    wavfile.write(filename='sample_clean.wav',rate=rate_x,data=signal[mask])
    # wavfile.write(filename='sample_clean.wav',rate=rate_x,data=signal)

# def calc_fft(y, rate):
#     n=len(y)
#     freq=np.fft.rfftfreq(n,d=1/rate)
#     Y=abs(np.fft.rfft(y)/n)
#     return (Y,freq)

# def audio_preprocessing():
#     signals={}
#     fft={}
#     fbank={}
#     mfccs={}
#     # down_sampling()
#     file='sample_clean.wav'
#     signal ,rate=librosa.load(file,sr=16000) 
#     # if signal.shape[0] is 0:
#     #     st.sidebar.warning("Please Record a valid audio")
#     #     return signals,fft,fbank,mfccs

#     signals[file]   = signal
#     fft[file]       = calc_fft(signal,rate)
#     fbank[file]     = logfbank(signal[:rate],rate,nfilt=26,nfft=1201).T  #nfft=sr/40
#     mfccs[file]     = mfcc(signal[:rate],rate,numcep=13,nfilt=26,nfft=1201).T  #nfft=sr/40

#     # st.sidebar.warning("procces  completed")
#     return  signals,fft,fbank,mfccs
    


def make_prediction():
    # rate, wav=wavfile.read('pos-0421-087-cough-f-40.wav')
    rate, wav=wavfile.read('pos-0421-087-cough-f-40.wav')

    y_prob=[]
    config.step=int(rate/10)
    for i in range(0,wav.shape[0]-config.step,config.step):
        sample=wav[i : i+ config.step]

        x=mfcc(sample,rate,numcep=config.nfeat,
                nfilt=config.nfilt,nfft=config.nfft)
        
        x = (x-config.min)/(config.max-config.min)

        if config.mode=='conv':
            x=x.reshape(1,x.shape[0],x.shape[1],1)
        elif config.mode=='time':
            x=x.reshape(1,x.shape[0],x.shape[1])

        y_hat=model.predict(x)
        y_prob.append(y_hat)

    x=np.mean(y_prob,axis=0).flatten()

    y_pred=(np.argmax(x))

    return y_pred









# signals={}
# fft={}
# fbank={}
# mfccs={}


classes=['covid', 'healthy']
p_path=os.path.join('pickles','conv.p')
with open(p_path,'rb') as handle:
    config = pickle.load(handle)
model=load_model(config.model_path)



# st.sidebar.header("COVID-19 DETECTION USING COUGH SIGNAL")
# st.sidebar.image("https://techcrunch.com/wp-content/uploads/2020/10/woman-coughing-covid-19-corona.jpg")
# # st.sidebar.title("")

# st.sidebar.write('press button to start recording')

# if st.sidebar.button("Record", type="primary"):
#     get_sample_audio()
    
#     # signal,sr=librosa.load("sample.wav")
#     # signals,fft,fbank,mfccs=audio_preprocessing()
    
#     down_sampling()

#     rate,signal=wavfile.read('sample_clean.wav')
#     if signal.shape[0] != 0: 

#         sr,signal=wavfile.read('sample.wav')
#         st.sidebar.audio(signal,sample_rate=sr)
        
#         if st.sidebar.button("Make Prediction"):
#             y_pred=make_prediction()
#             ans=classes[y_pred]
#             st.sidebar.title("")
#             st.sidebar.subheader(ans)
#     else:
#         st.sidebar.warning("Please Record a valid audio")



# # get_sample_audio()
# # signals,fft,fbank,mfccs=audio_preprocessing()

# time.sleep(5)


get_sample_audio()
down_sampling()
y_pred=make_prediction()
ans=classes[y_pred]
print('\n\n')
print(ans)
print('\n\n')





