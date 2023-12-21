import os

class Config:
    def __init__(self,mode='conv',nfilt=26,nfeat=13,nfft=512,rate=16000):
        self.mode=mode
        self.rate=rate
        self.nfft=nfft
        self.nfeat=nfeat
        self.nfilt=nfilt
        self.step=int(rate/10)
        self.model_path=os.path.join('models',mode +'.model')
        self.p_path=os.path.join('pickles',mode +'.p')






# Compute MFCC features from an audio signal.

# numcep – the number of cepstrum to return, default 13
# nfilt – the number of filters in the filterbank, default 26.
# nfft – the FFT size. Default is 512.
# signal – the audio signal from which to compute features. Should be an N*1 array



# for fast fourier transform   (fft)

# y=siganl
# n=len(y)
# freq=np.fft.rfftfreq(n,d=1/rate)
# Y=abs(np.fft.rfft(y)/n)
# return (Y,freq)


# Compute log Mel-filterbank energy features from an audio signal.

# nfilt – the number of filters in the filterbank, default 26.
# nfft – the FFT size. Default is 512.