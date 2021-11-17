import pandas as pd
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
from scipy.io import wavfile as wf
import os

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
        
                
    def remove_silences(self,index): #SAMPLE FUNCTION
        _,audio=wf.read(self.files[index])
        #do all processing here
        return output
        
    def __len__(self):
        return self.data_annotated.shape[0]
        
    def __getitem__(self, index):
        _,audio=wf.read(self.files[index])
        #directory = self.path_to_data + self.data_annotated.iloc[index,0]
        #filenames =  '{}/*.wav'.format(directory)
        #dirs_extracted = map(os.path.basename(self.path_to_data),glob.glob('{}/*.'.format(directory)))
        #audio=self.remove_silences(index) uncomment this when silence function is defined
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
    def __init__(self):
        pass

if __name__=='__main__':
    processed=preprocessed(path_to_csv='combined_data.csv',path_to_data='Extracted_Data')