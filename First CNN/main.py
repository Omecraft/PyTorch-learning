import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import tqdm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt  # Ajouté pour la visualisation

from model import MNISTCNN 
from dataset_loader import MNISTDataset 

# --- FONCTION DE VISUALISATION (LA NOUVELLE ACTION 'V') ---
def visualize_errors(train_ds, device, num_experts=10):
    print("\n🔍 Analyse des cas difficiles (Incertitude du Conseil des Sages)...")
    
    # On prend un échantillon de validation (le premier fold par exemple)
    indices = np.arange(len(train_ds))
    np.random.seed(42)
    np.random.shuffle(indices)
    val_idx = indices[:5000] # On analyse sur 5000 images
    val_loader = DataLoader(Subset(train_ds, val_idx), batch_size=1024, shuffle=False)

    all_expert_probs = []
    
    # 1. Collecte des probabilités de chaque expert
    for i in range(num_experts):
        path = f"expert_{i}.pth"
        if not os.path.exists(path): continue
        
        model = MNISTCNN().to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        
        expert_probs = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                probs = F.softmax(model(images), dim=1)
                expert_probs.append(probs.cpu())
        
        all_expert_probs.append(torch.cat(expert_probs))

    # 2. Calcul de la moyenne (Ensemble) et de l'incertitude
    ensemble_probs = torch.stack(all_expert_probs).mean(0)
    final_preds = torch.argmax(ensemble_probs, dim=1)
    
    # On récupère les vrais labels pour comparer
    actual_labels = torch.tensor([train_ds[i][1] for i in val_idx])
    
    # Calcul de la marge (Confiance 1er choix - Confiance 2e choix)
    sorted_probs, _ = torch.sort(ensemble_probs, dim=1, descending=True)
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    
    # On cherche les indices où la marge est la plus petite (grosse hésitation)
    hard_indices = torch.argsort(margins)[:10]

    # 3. Affichage Matplotlib
    plt.figure(figsize=(15, 7))
    plt.suptitle("Top 10 des images les plus litigieuses pour les Experts", fontsize=16)
    
    for i, idx in enumerate(hard_indices):
        plt.subplot(2, 5, i + 1)
        # On récupère l'image originale (déjà en tenseur via ton dataset)
        img = train_ds[val_idx[idx]][0].squeeze().numpy()
        
        pred = final_preds[idx].item()
        actual = actual_labels[idx].item()
        conf = ensemble_probs[idx][pred].item() * 100
        
        color = 'green' if pred == actual else 'red'
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {pred} | Vrai: {actual}\nConf: {conf:.1f}%", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    print("✅ Fenêtre d'affichage ouverte.")
    plt.show()

# --- TES FONCTIONS EXISTANTES (GARDÉES TELLES QUELLES) ---

def train_experts_kfold(train_ds, device, num_experts=10, epochs=50):
    # ... (ton code actuel reste identique)
    indices = np.arange(len(train_ds))
    np.random.seed(42) 
    np.random.shuffle(indices)
    folds = np.array_split(indices, num_experts)

    for i in range(num_experts):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(num_experts) if j != i])
        
        train_loader = DataLoader(
            Subset(train_ds, train_idx), batch_size=256, shuffle=True, 
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        
        val_loader = DataLoader(
            Subset(train_ds, val_idx), batch_size=1024, shuffle=False, 
            num_workers=0, pin_memory=True
        )

        model = MNISTCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

        best_val_acc = 0.0
        total_steps = epochs * len(train_loader)
        pbar = tqdm.tqdm(total=total_steps, desc=f"Expert {i+1}", unit="batch")

        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.update(1)

            model.eval()
            val_correct, val_total = 0, 0
            val_pbar = tqdm.tqdm(val_loader, desc="   ↳ Examen", leave=False)
            
            with torch.no_grad():
                for images, labels in val_pbar:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            pbar.set_postfix({"V_Acc": f"{val_acc:.2f}%", "Best": f"{best_val_acc:.2f}%"})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"expert_{i}.pth")
            
            scheduler.step()
        pbar.close()

def predict_ensemble_tta(num_experts, device):
    # ... (ton code actuel de prédiction TTA reste identique)
    print("\n🚀 PRÉDICTION AVEC TTA...")
    test_ds = MNISTDataset('./data/test.csv', is_test=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)
    
    tta_transforms = [
        lambda x: x,
        lambda x: transforms.functional.rotate(x, 5),
        lambda x: transforms.functional.rotate(x, -5),
        lambda x: transforms.functional.affine(x, 0, (1, 1), 1, 0),
        lambda x: transforms.functional.affine(x, 0, (-1, -1), 1, 0)
    ]

    all_expert_scores = None 
    for i in range(num_experts):
        path = f"expert_{i}.pth"
        if not os.path.exists(path): continue
        model = MNISTCNN().to(device); model.load_state_dict(torch.load(path)); model.eval()
        
        expert_logits_list = []
        with torch.no_grad():
            for images in tqdm.tqdm(test_loader, desc=f"Expert {i+1} vote"):
                images = images.to(device)
                batch_tta_probs = torch.zeros((images.size(0), 10)).to(device)
                for t in tta_transforms:
                    augmented = torch.stack([t(img) for img in images])
                    batch_tta_probs += F.softmax(model(augmented), dim=1)
                expert_logits_list.append(batch_tta_probs.cpu() / len(tta_transforms))
        
        expert_avg_probs = torch.cat(expert_logits_list)
        all_expert_scores = expert_avg_probs if all_expert_scores is None else all_expert_scores + expert_avg_probs

    _, final_preds = torch.max(all_expert_scores, 1)
    pd.DataFrame({"ImageId": range(1, len(final_preds) + 1), "Label": final_preds.numpy()}).to_csv("submission_final.csv", index=False)
    print("\n🏆 Submission prête !")

# --- MAIN MIS À JOUR ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_experts = 10
    
    choix = input("Action : (e)ntraîner, (p)rédire ou (v)isualiser erreurs : ").lower()

    # On charge le dataset sans augmentation pour la validation/visualisation
    # On laisse les transfos optionnelles pour l'entraînement 'e'
    if choix == 'e':
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
        ])
        train_ds = MNISTDataset('./data/train.csv', transform=train_transform)
        train_experts_kfold(train_ds, device, num_experts=num_experts, epochs=50)
        predict_ensemble_tta(num_experts, device)
    
    elif choix == 'v':
        # On charge sans transform pour bien voir l'image originale
        train_ds = MNISTDataset('./data/train.csv', transform=None)
        visualize_errors(train_ds, device, num_experts=num_experts)
        
    elif choix == 'p':
        predict_ensemble_tta(num_experts, device)

if __name__ == "__main__":
    main()