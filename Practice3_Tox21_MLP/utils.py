import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE




def read_data(filename):
    f = open('../Data/tox21/'+filename + '.smiles', 'r')
    contents = f.readlines()

    smiles = []
    labels = []
    for i in contents:
        smi = i.split()[0]
        label = int(i.split()[2].strip())

        smiles.append(smi)
        labels.append(label)

    num_total = len(smiles)
    rand_int = np.random.randint(num_total, size=(num_total,))
    
    return np.asarray(smiles)[rand_int], np.asarray(labels)[rand_int]


# Generate bit vector from given smile string
def get_fingerprint(smile, args):
    mol = Chem.MolFromSmiles(smile)
    fingerprint = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=args.nbits)
    bit_string = fingerprint.ToBitString()
    bit_vec = np.array([int(bit) for bit in bit_string])
    return bit_vec
    
    
class myDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    
# Generate Dataset by shuffling, fingerprinting, spliting
def make_dataset(args):
    smiles, label = read_data(args.tox_type)
    smiles, label = shuffle(smiles, label, random_state=args.seed)
    
    # Convert Smiles to Morgan Fingerprints and exclude failure cases
    list_vec = list()
    list_label = list()
    for i, smile in enumerate(smiles):
        try:
            vec = get_fingerprint(smile, args)
            list_vec.append(vec)
            list_label.append(label[i])
        except:
            pass
    
    return np.array(list_vec), np.array(list_label)


def make_partition(X, y, args):
    list_fold = list()
    cv = StratifiedKFold(n_splits=args.n_splits, random_state=args.seed)
    for train_index, test_index in cv.split(X, y):
        # Spliting data into train & test set 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Spliting train data into train & validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                          y_train, 
                                                          test_size=args.test_size,
                                                          random_state=args.seed)
        # Handling class imbalnce problem by oversampling toxic data using SMOTE algorithm 
        sm = SMOTE(ratio='auto', kind='regular')
        X_train, y_train = sm.fit_sample(X_train, y_train)

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