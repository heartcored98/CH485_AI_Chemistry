import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold

    
def read_ZINC(num_mol):
    f = open('../Data/logP/ZINC.smiles', 'r')
    contents = f.readlines()

    list_smi = []
    fps = []
    logP = []
    tpsa = []
    for i in range(num_mol):
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


class myDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
def make_partition(args):
    smiles, logPs, _ = read_ZINC(args.num_mol)
    
    # Truncate smile strings
    for i, smile in enumerate(smiles):
        truncated_smile = smile[:args.max_len]
        filled_smile = truncated_smile + ' '* (args.max_len-len(truncated_smile))
        smiles[i] = filled_smile
        
    X, y = np.array(smiles), np.array(logPs)

    list_fold = list()
    cv = KFold(n_splits=args.n_splits, random_state=args.seed)
    for train_index, test_index in cv.split(X, y):
        # Spliting data into train & test set 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Spliting train data into train & validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                          y_train, 
                                                          test_size=args.test_size,
                                                          random_state=args.seed)

        # Construct dataset object
        train_set = myDataset(X_train, y_train)
        val_set = myDataset(X_val, y_val)
        test_set = myDataset(X_test, y_test)

        partition = {
            'train': train_set,
            'val': val_set,
            'test': test_set
        }
        list_fold.append(partition)
    return list_fold