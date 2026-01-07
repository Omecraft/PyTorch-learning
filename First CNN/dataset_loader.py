import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, csv_file, is_test=False, transform=None):
        print(f"⌛ Chargement et conversion de {csv_file}...")
        df = pd.read_csv(csv_file)
        self.is_test = is_test
        self.transform = transform
        
        if not is_test:
            # On stocke tout en Tenseur directement
            self.y = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
            self.x = torch.tensor(df.iloc[:, 1:].values.astype('float32') / 255.0)
        else:
            self.x = torch.tensor(df.values.astype('float32') / 255.0)
            self.y = None
        
        del df # Libère la mémoire
        print("✅ Données prêtes en RAM.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # On récupère le tenseur (C, H, W)
        image = self.x[idx].reshape(1, 28, 28)
        
        # On applique la transformation directement sur le tenseur
        if self.transform:
            image = self.transform(image)
            
        if not self.is_test:
            return image, self.y[idx]
        return image