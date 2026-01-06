import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# On importe TES classes depuis TES fichiers
from dataset import MNISTDataset 
from model import MNISTModel

def main():
    # 1. Configuration du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Chargement des données
    train_ds = MNISTDataset('./data/train.csv')
    
    # Le DataLoader : il utilise ton dataset pour créer des paquets
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    # Test : on regarde un batch
    images, labels = next(iter(train_loader))
    print(f"Forme du batch d'images : {images.shape}") # [64, 1, 28, 28]
    print(f"Forme du batch de labels : {labels.shape}") # [64]

if __name__ == "__main__":
    main()