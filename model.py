import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import librosa
from python_speech_features import mfcc ,logfbank
from scipy.io import wavfile
# import csv
from tqdm import tqdm
# import tensorflow as tf
# import keras
from keras.utils import to_categorical 
from keras.layers import *
from keras.models import  Sequential
# from sklearn.utils.class_weight import compute_class_weight
import pickle
from keras.callbacks import ModelCheckpoint
from cfg import Config
from sklearn.model_selection import train_test_split
def check_data():
    if os.path.isfile(config.p_path):
        print('Loading existing data for {} model'.format(config.mode))
        with open(config.p_path,'rb') as f:
            tmp=pickle.load(f)
            return tmp
    else:
        return None

def built_rand_feat():
    tmp=check_data()
    if tmp:
        return tmp.data[0],tmp.data[1]
    X=[]
    y=[]
    _min,_max=float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class=np.random.choice(class_dist.index,p=prob_dist)
        file=np.random.choice(df[df['label']==rand_class].index)
        rate,wav=wavfile.read('clean/' + file)
        # label=rand_class
        label=df.at[file,'label']
        rand_index=np.random.randint(0,wav.shape[0]-config.step)
        sample=wav[rand_index:rand_index+config.step]
        X_sample=mfcc(sample,rate,numcep=config.nfeat,
                      nfilt=config.nfilt,nfft=config.nfft)
        _min=min(np.amin(X_sample),_min)
        _max=max(np.amax(X_sample),_max)
        # X.append(X_sample if config.mode=='conv' else x_sample.T)
        X.append(X_sample)
        y.append(classes.index(label))
    config.min=_min
    config.max=_max
    X,y=np.array(X),np.array(y)
    X=(X - _min)/(_max-_min)  # to normalise the data 

    if config.mode=='conv2':
        X=X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    elif config.mode=="time":
        X=X.reshape(X.shape[0],X.shape[1],X.shape[2])

    y=to_categorical(y,num_classes=2)

    config.data=(X,y)
    with open(config.p_path,'wb') as f:
        pickle.dump(config,f,protocol=2)

    return X,y



def get_conv_model():
    model=Sequential()
    model.add(Conv2D( 16 ,kernel_size=(3,3),activation="relu",strides=(1,1),
                     padding='same',input_shape=input_shape))
    model.add(Conv2D( 32 ,kernel_size=(3,3),activation="relu",strides=(1,1),
                     padding='same'))
    model.add(Conv2D( 64 ,kernel_size=(3,3),activation="relu",strides=(1,1),
                     padding='same'))
    model.add(Conv2D( 128 ,kernel_size=(3,3),activation="relu",strides=(1,1),
                     padding='same'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(2,activation='sigmoid'))
    model.summary()
    model.compile( loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

def get_recurrent_model():
    model=Sequential()
    model.add(LSTM(128,return_sequences=True,input_shape=input_shape))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64,activation='relu')))
    model.add(TimeDistributed(Dense(32,activation='relu')))
    model.add(TimeDistributed(Dense(16,activation='relu')))
    model.add(TimeDistributed(Dense(8,activation='relu')))
    model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(2,activation='sigmoid'))
    model.summary()
    model.compile( loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model



df=pd.read_csv('audio_without_pitch.csv')

y=df['label']
x=df.drop('label',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
train=pd.concat( [x_train,y_train],axis=1,)
train.reset_index(drop=True,inplace=True)

df=train


df.set_index('filename',inplace=True)

for f in df.index:
    rate,signal=wavfile.read('clean/'+f)
    df.at[f,'length']=signal.shape[0]/rate
classes=list(np.unique(df.label))
class_dist=df.groupby(['label'])['length'].mean()

n_samples=4* int(df['length'].sum()/.1)
prob_dist= class_dist/class_dist.sum()
choices=np.random.choice(class_dist.index,p=prob_dist)

# fig ,ax=plt.subplots()
# ax.set_title("Class Distribution",y=1.08)
# ax.pie(class_dist,labels=class_dist.index,autopct='%1.1f%%',
#        shadow=False,startangle=90)
# ax.axis('equal')
# plt.show()




config=Config(mode='conv2')

if config.mode=='conv2':
    X,y=built_rand_feat()
    y_flat=np.argmax(y,axis=1)
    input_shape=X.shape[1:]
    model=get_conv_model()


elif config.mode=="time":
    X,y=built_rand_feat()
    y_flat=np.argmax(y,axis=1)
    input_shape=X.shape[1:]
    model=get_recurrent_model()

# class_weight=compute_class_weight('balanced',
#                                   classes=np.unique(y_flat),y=y_flat)
# class_weight=class_weight,
# checkpoint=ModelCheckpoint(config.model_path,monitor='val_acc',verbose=1,mode='max',
#                             save_best_only=True,save_weights_only=False,period=1)



# model.fit(X,y,epochs=20,batch_size=32,
#           shuffle=True,validation_split=0.1,callbacks=[checkpoint])
# model.save(config.model_path)

