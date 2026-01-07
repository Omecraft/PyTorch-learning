import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, csv_file, is_test=False):
        # 1. On charge le CSV avec Pandas
        df = pd.read_csv(csv_file)
        
        self.is_test = is_test
        
        if not is_test:
            # Pour l'entraînement : la colonne 0 est le chiffre (le label)
            self.y = df.iloc[:, 0].values
            # Les autres colonnes sont les 784 pixels
            self.x = df.iloc[:, 1:].values.astype('float32')
        else:
            # Pour le test Kaggle : il n'y a pas de colonne label
            self.x = df.values.astype('float32')
            self.y = None
            
        # 2. Normalisation (comme tu l'avais fait en NumPy)
        # On passe de [0, 255] à [0, 1]
        self.x = self.x / 255.0

    def __len__(self):
        # On dit à PyTorch combien il y a d'images au total
        return len(self.x)

    def __getitem__(self, idx):
        # C'est ici qu'on récupère UNE image précise
        
        # On transforme la ligne de pixels en Tenseur PyTorch
        # Note : On reshape en (1, 28, 28) car c'est une image Noir & Blanc (1 canal)
        image = torch.tensor(self.x[idx]).reshape(1, 28, 28)
        
        if not self.is_test:
            label = torch.tensor(self.y[idx], dtype=torch.long)
            return image, label
        else:
            return image