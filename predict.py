from keras.models import load_model
from scipy.io import wavfile
from python_speech_features import mfcc ,logfbank
from sklearn.metrics import accuracy_score
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.metrics import roc_curve, auc,roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



def make_prediction(audio_dir='clean/'):

    y_true=[]
    y_pred=[]
    fn_prob={}
    
    print('Extracting features from audio')
    # for fn in tqdm(os.listdir(audio_dir)):
    for _,row in tqdm(df.iterrows()):
        fn=row[0]
        rate, wav=wavfile.read(os.path.join(audio_dir,fn))
        label=fn2class[fn]
        c=classes.index(label)
        y_prob=[]
        for i in range(0,wav.shape[0]-config.step,config.step):
            sample=wav[i : i+ config.step]
            x=mfcc(sample,rate,numcep=config.nfeat,
                    nfilt=config.nfilt,nfft=config.nfft)
            x = (x-config.min)/(config.max-config.min)

            if config.mode=='conv2':
                x=x.reshape(1,x.shape[0],x.shape[1],1)
            elif config.mode=='time':
                x=x.reshape(1,x.shape[0],x.shape[1])
                # x=np.expand_dims(x,axis=0)
            y_hat=model.predict(x)
            y_prob.append(y_hat)
            # y_pred.append(np.argmax(y_hat))
        y_true.append(c)

        fn_prob[fn]=np.mean(y_prob,axis=0).flatten()
        y_pred.append(np.argmax(fn_prob[fn]))

      
        
    return y_true,y_pred,fn_prob 

def plot_actual_vs_predicted(y_true,y_pred,title,grafik='yüzde',cmp='cividis'):
    cm = confusion_matrix(y_true,y_pred)
    cm=cm.astype(np.double)
    #cm=cm/np.sum(cm)

    plt.figure(figsize=(5,5))
    if (grafik=='yüzde'):
      cm=(np.round(cm / cm.sum(axis=1),4))#*100
      sns.heatmap(cm,annot=True,fmt='.2%',xticklabels=classes,yticklabels=classes) #fmt='g',
    else:
      sns.heatmap(cm,annot=True,fmt='g',xticklabels=classes,yticklabels=classes)
    plt.title(title)
    plt.show()
    print("Classification Report")
    print(classification_report(y_true,y_pred))   



df=pd.read_csv('audio_without_pitch.csv')

y=df['label']
x=df.drop('label',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
test=pd.concat( [x_test,y_test],axis=1,)
test.reset_index(drop=True,inplace=True)

df=test

classes=list(np.unique(df.label))
fn2class={}
for i,row in df.iterrows():
    fn2class[row[0]]=row[1]
p_path=os.path.join('pickles','conv2.p')
with open(p_path,'rb') as handle:
    config = pickle.load(handle)
model=load_model(config.model_path)

y_true,y_pred,fn_prob=make_prediction('clean/')
# acc_score=accuracy_score(y_true=y_true,y_pred=y_pred)
# y_probs=[]
# for i,row in df.iterrows():
#     y_prob=fn_prob[row.filename]
#     y_probs.append(y_prob)
#     for c,p in zip(classes,y_prob):
#         df.at[i,c]=p

# y_pred_class=[classes[np.argmax(y)] for y in y_probs]
# df['y_pred_class']=y_pred_class
# df.to_csv('predictions_pitched.csv',index=False)



plot_actual_vs_predicted(y_true,y_pred,"Test Data Predictions",'normal')
