import multiprocessing as mp
import os
from os.path import isfile, join
import time
import argparse
from random import sample
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.rdMolDescriptors import CalcTPSA #,CalcPBF
import pandas as df
import re
import numpy as np
from numpy.random import choice
import gc

def process_smile(row):
    """Return molecular properties """
    try:
        smi = row.strip()
        m = Chem.MolFromSmiles(smi)
        logP = MolLogP(m)
        length = len(list(smi))

        del m
        return smi, logP, length
    except:
        return None, None, None

if __name__ == '__main__':
    num_worker = 14
    with open('ZINC.smiles') as file:
        list_row = file.readlines()[1:]

        with mp.Pool(processes=num_worker) as pool:
            data = pool.map(process_smile, list_row)

        train_data, val_data = train_test_split(data, test_size=0.2)
        label_columns = ['smile', 'logP', 'length']
        df_train = df.DataFrame.from_records(train_data[:60000], columns=label_columns)
        df_train = df_train.dropna()
        df_train.sort_values(by=['length'], ascending=False, inplace=True)
        df_train.to_csv(path_or_buf='train_logP.csv',
                        float_format='%g', index=False)

        df_val = df.DataFrame.from_records(val_data[:10000], columns=label_columns)
        df_val = df_val.dropna()
        df_val.sort_values(by=['length'], ascending=False, inplace=True)
        df_val.to_csv(path_or_buf='val_logP.csv',
                      float_format='%g', index=False)


