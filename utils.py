import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import glob
import numpy as np
from scipy.io import wavfile as wf
import librosa
from scipy.stats import kurtosis
from python_speech_features import logfbank
import os
import math


class preprocessed(Dataset):
    '''
    Load the audio data, and Normalize, Remove Silences, balance (if required) and apply any/other preprocessing/transformation on it.    
    
    preprocess.get_processed(:params)-
    applies all the required transformations and 
    either returns the data as a tensor or stores as new audio files in some directory..
    '''
    def __init__(self,path_to_csv,device=None,transform=None,path_to_data='same'):
        
        self.path_to_csv = path_to_csv
        if path_to_data == 'same':
            self.path_to_data=path_to_csv + '/..'
        else:
            self.path_to_data=path_to_data + '/' if path_to_data[-1] !='/' else path_to_data        
        data_annotated = pd.read_csv(path_to_csv)
        
        self.data_annotated=pd.DataFrame(np.zeros(data_annotated.shape),columns=list(data_annotated.columns))
        self.files=glob.glob(self.path_to_data+'*/*/*.wav')
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transform
        
                
    def remove_silences(self,index): 
        time_win = 0.05
        amp_thresh = 0.005
        sample_rate = 44100        
        
        # Loading audio
        signal, sr = torchaudio.load(self.files[index])

        # Resampling
        signal_rs = torchaudio.transforms.Resample(sr, 44100) 
        signal_rs = signal_rs(signal)
        signal_length = signal_rs.size()[1]

        # Normalizing
        signal_nr = 0.9 * (signal_rs/(signal_rs.max())) # c_i(t)

        # Equation 2
        signal_I = []
        for j in range(0, math.floor(signal_length/(44100*time_win))):
            j_mu_lam = j * 44100 * time_win
            j1_mu_lam = (j+1) * 44100 * time_win # (j+1)_mu_lam
            signal_I.append(signal_nr[0, int(j_mu_lam) : int(j1_mu_lam)])

        sig_I = torch.stack(signal_I)
        sig_I = torch.flatten(sig_I) # (393, 2205)
        sig_I = torch.unsqueeze(sig_I, 0) #C_I(t)

        # Equation 3
        silenced_signal = []
        for j in range(sig_I.size()[1]):
            if np.abs(sig_I[0,j]) >= amp_thresh:
                silenced_signal.append(sig_I[0,j])
                
        silenced_signal = torch.tensor(silenced_signal)
        silenced_signal = torch.unsqueeze(silenced_signal, 0) 

        return silenced_signal       
        
    def __len__(self):
        return self.data_annotated.shape[0]
        
    def __getitem__(self, index):
        _,audio=wf.read(self.files[index])
        #directory = self.path_to_data + self.data_annotated.iloc[index,0]
        #filenames =  '{}/*.wav'.format(directory)
        #dirs_extracted = map(os.path.basename(self.path_to_data),glob.glob('{}/*.'.format(directory)))
        audio=self.remove_silences(index) 
        return audio
    
    def get_processed(self):
        '''
        Returns the audio data with all the processings/transformations applied
        '''
        pass
    
    
    
class extract_features:
    '''
    Extracts all of the features from the processed audio-files.
    '''
    def __init__(self,path_to_csv,device=None,transform=None,path_to_data='same'):
        
        self.path_to_csv = path_to_csv
        if path_to_data == 'same':
            self.path_to_data=path_to_csv + '/..'
        else:
            self.path_to_data=path_to_data + '/' if path_to_data[-1] !='/' else path_to_data        
        data_annotated = pd.read_csv(path_to_csv)
        
        self.data_annotated=pd.DataFrame(np.zeros(data_annotated.shape),columns=list(data_annotated.columns))
        self.files=glob.glob(self.path_to_data+'*/*/*.wav')
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transform
    

    #kurtosis
    def kurtosis(self, index):
        signal, sr = librosa.load(self.files[index])
        kurtosis_var = kurtosis(signal)
        return kurtosis_var
    
    #log Mel-filterbank energy feature
    def log_energy(self, index):
        signal, sr = librosa.load(self.files[index])
        log_val = logfbank(signal, sr)
        return log_val
    
    #Zero crossing rate
    def zcr(self,index):
        FRAME_LENGTH=4096
        HOP_LENGHT=2048
        
        signal = np.array(signal)
        signal = signal.reshape((len(signal[0,:]),))
        zcr_signal = librosa.feature.zero_crossing_rate(signal, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
        return zcr_signal

if __name__=='__main__':
    processed=preprocessed(path_to_csv='combined_data.csv',path_to_data='Extracted_Data')
    feature_extraction = extract_features(path_to_csv='combined_data.csv',path_to_data='Extracted_Data')
