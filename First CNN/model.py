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
        
        # 3. Le MaxPool : Il divise la taille par 2 en ne gardant que le pixel le plus fort
        # Ça permet de dire : "La boucle est là, peu importe sa position exacte"
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        # --- PARTIE 2 : CLASSIFICATEUR (LINEAR) ---
        
        # Après 2 MaxPool, l'image 28x28 est devenue 7x7 (28 -> 14 -> 7)
        # On a 64 filtres de taille 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x arrive en (Batch, 1, 28, 28)
        
        # Bloc 1 : Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # Devient (Batch, 32, 14, 14)
        
        # Bloc 2 : Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # Devient (Batch, 64, 7, 7)
        
        # On "aplatit" (Flatten) seulement ici pour entrer dans les couches Linear
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x