import pandas as pd
import torchaudio 
import torch

class preprocess:
    '''
    Load the audio data, and Normalize, Remove Silences, balance (if required) and apply any/other preprocessing/transformation on it.    
    
    preprocess.get_processed(:params)-
    applies all the required transformations and 
    either returns the data as a tensor or stores as new audio files in some directory..
    '''
    def __init__(self):
        pass
    
    
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

