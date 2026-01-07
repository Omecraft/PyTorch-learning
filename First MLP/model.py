import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # On définit les ingrédients ici
        # 1. On "aplatit" l'image (28x28 pixels -> 784 neurones d'entrée)
        self.flatten = nn.Flatten()
        
        # 2. Première couche : 784 entrées -> 128 neurones (couche cachée)
        self.fc1 = nn.Linear(784, 128)
        
        # 3. Deuxième couche : 128 neurones -> 64 neurones
        self.fc2 = nn.Linear(128, 64)
        
        # 4. Couche de sortie : 64 neurones -> 10 neurones (un pour chaque chiffre de 0 à 9)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        # On aplatit l'image
        x = self.flatten(x)
        
        # Passage dans la couche 1 + Activation ReLU
        # ReLU est plus rapide et efficace que la Sigmoïde pour les réseaux profonds
        x = F.relu(self.fc1(x))
        
        # Passage dans la couche 2 + Activation ReLU
        x = F.relu(self.fc2(x))
        
        # Passage dans la couche de sortie
        # Pas d'activation ici car on utilisera une fonction d'erreur (Loss) 
        # qui s'en occupe pour nous
        x = self.fc3(x)
        
        return x