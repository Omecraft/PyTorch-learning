import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        
        # --- PARTIE 1 : EXTRACTEUR DE FORMES (CONVOLUTION) ---
        
        # 1. On cherche 32 motifs simples (lignes, bords) avec des filtres 3x3
        # in_channels=1 (Noir et blanc), out_channels=32 filtres
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 2. On cherche 64 motifs complexes (boucles, angles)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 3. On cherche 128 motifs encore plus complexes
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Le MaxPool : Il divise la taille par 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        # --- PARTIE 2 : CLASSIFICATEUR (LINEAR) ---
        
        # Après 3 MaxPools : 28 -> 14 -> 7 -> 3
        # On a 128 filtres de taille 3x3
        self.fc1 = nn.Linear(128 * 3 * 3, 256) # Augmenté à 256 neurones
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # x arrive en (Batch, 1, 28, 28)
        
        # Bloc 1 : Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # Devient (Batch, 32, 14, 14)
        
        # Bloc 2 : Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # Devient (Batch, 64, 7, 7)
        
        # Bloc 3 : Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # Devient (Batch, 128, 3, 3)
        
        # On "aplatit" (Flatten) seulement ici pour entrer dans les couches Linear
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x