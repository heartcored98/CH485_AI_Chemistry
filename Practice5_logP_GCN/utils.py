import os
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from torch.utils.data import Dataset
from scipy.linalg import fractional_matrix_power
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm_notebook

"""
Normalization Ref: https://tkipf.github.io/graph-convolutional-networks/
"""

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.diag(np.array(mx.sum(1)))
    r_inv = np.linalg.matrix_power(rowsum, -1)
    r_inv[np.isinf(r_inv)] = 0.
    return r_inv.dot(mx)

def normalize_adj(mx):
    """Symmetry Normalization"""
    rowsum = np.diag(np.array(mx.sum(1)))
    r_inv = fractional_matrix_power(rowsum, -0.5)
    r_inv[np.isinf(r_inv)] = 0.
    return r_inv.dot(mx).dot(r_inv)

def read_ZINC(num_mol):
    f = open('../Data/logP/ZINC.smiles', 'r')
    contents = f.readlines()

    list_smi = []
    fps = []
    logP = []
    tpsa = []
    for i in tqdm_notebook(range(num_mol)):
        smi = contents[i].strip()
        list_smi.append(smi)
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m,2)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp,arr)
        fps.append(arr)
        logP.append(MolLogP(m))
        tpsa.append(CalcTPSA(m))

    fps = np.asarray(fps).astype(float)
    logP = np.asarray(logP).astype(float)
    tpsa = np.asarray(tpsa).astype(float)

    return list_smi, logP, tpsa

def convert_to_graph(smiles_list):
    adj = []
    adj_norm = []
    features = []
    maxNumAtoms = 50
    for i in tqdm_notebook(smiles_list):
        # Mol
        iMol = Chem.MolFromSmiles(i.strip())
        #Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        if( iAdjTmp.shape[0] <= maxNumAtoms):
            # Feature-preprocessing
            iFeature = np.zeros((maxNumAtoms, 19))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append( atom_feature(atom) ) ### atom features only
            iFeature[0:len(iFeatureTmp), 0:19] = iFeatureTmp ### 0 padding for feature-set
            features.append(iFeature)

            # Adj-preprocessing & Symmetrically normalizing the adj matrix.
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] =  normalize_adj(iAdjTmp + np.eye(len(iFeatureTmp)))
#             iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(np.asarray(iAdj))
    features = np.asarray(features)
    adjs = np.array(adj)
    return features, adjs
    
def atom_feature(atom):
    return np.array(char_to_ix(atom.GetSymbol(),
                              ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                               'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                               'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                               'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (40, 6, 5, 6, 1)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def char_to_ix(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [allowable_set.index(x)+1]
    


class myDataset(Dataset):
    def __init__(self, X1, X2, y):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        
    def __len__(self):
        return len(self.X1)
    
    def __getitem__(self, index):
        return self.X1[index], self.X2[index], self.y[index]
    
def make_partition(num_mol, val_size, test_size, seed):
    smiles, logPs, _ = read_ZINC(num_mol)
    
    X, y = np.array(smiles), np.array(logPs)
    features, adjs = convert_to_graph(X)

    # Spliting train data into train & validation set
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(features, adjs, y, 
                                                                             test_size=test_size,
                                                                             random_state=seed)
    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1_train, X2_train, y_train, 
                                                                           test_size=val_size,
                                                                           random_state=seed)

    # Construct dataset object
    train_set = myDataset(X1_train, X2_train, y_train)
    val_set = myDataset(X1_val, X2_val, y_val)
    test_set = myDataset(X1_test, X2_test, y_test)

    partition = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }
       
    return partition

from decimal import Decimal
import json 
from os import listdir
from os.path import isfile, join
import pandas as pd
import hashlib

class Writer():
    
    def __init__(self, prior_keyword=[], dir='./results'):
        self.prior_keyword = prior_keyword
        self.dir = dir
        
    def generate_hash(self, args):
        str_as_bytes = str.encode(str(args))
        hashed = hashlib.sha256(str_as_bytes).hexdigest()[:24]
        return hashed

    def write(self, args, prior_keyword=None):
        dict_args = vars(args)
        if 'bar' in dict_args:
            del dict_args['bar']
        if prior_keyword:
            self.prior_keyword = prior_keyword
        filename = 'exp_{}'.format(args.exp_name)
        for keyword in self.prior_keyword:
            value = str(dict_args[keyword])
            if value.isdigit():
                filename += keyword + ':{:.2E}_'.format(Decimal(dict_args[keyword]))
            else:
                filename += keyword + ':{}_'.format(value)
#         hashcode = self.generate_hash(args)
#         filename += hashcode
        filename += '.json'
        
        with open(self.dir+'/'+filename, 'w') as outfile:
            json.dump(dict_args, outfile)
            
    def read(self, exp_name=''):
        list_result = list()
        filenames = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        for filename in filenames:
            with open(join(self.dir, filename), 'r') as infile:
                result = json.load(infile)
                if len(exp_name) > 0:
                    if result['exp_name'] == exp_name:
                        list_result.append(result)
                else:
                    list_result.append(result)
                        
        return pd.DataFrame(list_result)
    
    def clear(self, exp_name=''):
        filenames = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        for filename in filenames:
            if len(exp_name) > 0:
                result = json.load(open(join(self.dir, filename), 'r'))
                if result['exp_name'] == exp_name:
                    os.remove(join(self.dir, filename))
            else:
                os.remove(join(self.dir, filename))