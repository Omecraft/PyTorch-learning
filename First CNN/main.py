import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm
import pandas as pd
import os

from model import MNISTCNN 
from dataset_loader import MNISTDataset 

def train_one_expert(expert_id, train_loader, device, epochs=40):
    print(f"\n" + "="*50)
    print(f"🎓 ENTRAÎNEMENT DE L'EXPERT N°{expert_id+1}")
    print("="*50)
    
    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    total_batches = len(train_loader)
    total_steps = epochs * total_batches
    
    pbar = tqdm.tqdm(total=total_steps, desc=f"Expert {expert_id+1}", unit="batch")

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.update(1)
            pbar.set_postfix({
                "Ep": f"{epoch+1}/{epochs}",
                "Acc": f"{(100 * correct / total):.2f}%"
            })
        
        scheduler.step()
    
    pbar.close()
    torch.save(model.state_dict(), f"expert_{expert_id}.pth")
    return model

def predict_ensemble_tta(num_experts, device):
    print("\n" + "🚀"*5 + " PRÉDICTION AVEC TTA (50 AVIS PAR IMAGE) " + "🚀"*5)
    
    test_ds = MNISTDataset('./data/test.csv', is_test=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    
    # 5 transformations TTA pour stabiliser la prédiction
    tta_transforms = [
        lambda x: x,                                      # Original
        lambda x: transforms.functional.rotate(x, 5),     # +5°
        lambda x: transforms.functional.rotate(x, -5),    # -5°
        lambda x: transforms.functional.affine(x, 0, (1, 1), 1, 0),  # Décalage léger
        lambda x: transforms.functional.affine(x, 0, (-1, -1), 1, 0) # Décalage opposé
    ]

    all_expert_scores = None 

    for i in range(num_experts):
        path = f"expert_{i}.pth"
        if not os.path.exists(path):
            print(f"⚠️ Fichier {path} manquant, expert ignoré.")
            continue

        print(f"Expert {i+1} analyse avec TTA...")
        model = MNISTCNN().to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        
        expert_logits_list = []
        with torch.no_grad():
            for images in tqdm.tqdm(test_loader, leave=False):
                images = images.to(device)
                batch_tta_probs = torch.zeros((images.size(0), 10)).to(device)
                
                for t in tta_transforms:
                    augmented = torch.stack([t(img) for img in images])
                    outputs = model(augmented)
                    batch_tta_probs += F.softmax(outputs, dim=1)
                
                expert_logits_list.append(batch_tta_probs.cpu() / len(tta_transforms))
        
        expert_avg_probs = torch.cat(expert_logits_list)
        if all_expert_scores is None:
            all_expert_scores = expert_avg_probs
        else:
            all_expert_scores += expert_avg_probs

    _, final_preds = torch.max(all_expert_scores, 1)
    df = pd.DataFrame({"ImageId": range(1, len(final_preds) + 1), "Label": final_preds.numpy()})
    df.to_csv("submission_ensemble_tta.csv", index=False)
    print("\n🏆 Fichier 'submission_ensemble_tta.csv' prêt !")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_experts = 10
    
    print(f"Utilisation du matériel : {device}")
    choix = input("Que voulez-vous faire ? (e)ntraîner les experts ou (p)rédire : ").lower()

    if choix == 'e':
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
        ])
        train_ds = MNISTDataset('./data/train.csv', transform=train_transform)
        train_loader = DataLoader(
            train_ds, batch_size=256, shuffle=True, 
            num_workers=8, pin_memory=True, persistent_workers=True
        )

        for i in range(num_experts):
            train_one_expert(i, train_loader, device, epochs=40)
        
        # On peut lancer la prédiction directement après l'entraînement
        predict_ensemble_tta(num_experts, device)

    elif choix == 'p':
        predict_ensemble_tta(num_experts, device)
    else:
        print("Choix invalide. Tapez 'e' ou 'p'.")

if __name__ == "__main__":
    main()