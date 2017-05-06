#!/usr/bin/env python

import os
from lm_lstm_timit import train

def main(job_id, params):
    print(params)
    validerr = train(
        saveto=params['model'][0],
        reload_=params['reload'][0],
        dim_input=params['dim_input'][0],
        dim=params['dim'][0],
        #n_words=params['n-words'][0],
        decay_c=params['decay-c'][0],
        lrate=params['learning-rate'][0],
        optimizer=params['optimizer'][0],
        maxlen=40,
        batch_size=32,
        valid_batch_size=32,
        validFreq=2000,
        dispFreq=10,
        saveFreq=1000,
        sampleFreq=1000,
        dataset=None,
        valid_dataset=None,
        dictionary=None,
        use_dropout=params['use-dropout'][0],
        kl_start = 0.2,
        kl_rate =  0.00005)
    return validerr

if __name__ == '__main__':
    try:
        # Created experiments folder, if needed.
        os.makedirs("./experiments/timit/")
    except:
        pass

    main(0, {
        'model': ['./experiments/timit/'],
        'dim_input': [200],
        'dim': [1000],
        #'n-words': [50],
        'optimizer': ['adam'],
        'decay-c': [0.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})
