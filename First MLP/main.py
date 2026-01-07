import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import pandas as pd

# On importe TES classes depuis TES fichiers
from First MLP.dataset_loader import MNISTDataset 
from First MLP.model import MNISTModel

def main():
    # 1. Configuration du device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Force le CPU le temps de régler le problème de driver
    device = torch.device("cpu")
    
    # 2. Chargement des données
    train_ds = MNISTDataset('./data/train.csv')
    
    # Le DataLoader : il utilise ton dataset pour créer des paquets
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    # Test : on regarde un batch
    images, labels = next(iter(train_loader))
    print(f"Forme du batch d'images : {images.shape}") # [64, 1, 28, 28]
    print(f"Forme du batch de labels : {labels.shape}") # [64]

    # 3. Création du modèle
    model = MNISTModel().to(device)
    
    # 4. Le jugement
    criterion = nn.CrossEntropyLoss()

    # 5. L'optimisation
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 6. L'entraînement ---
    epochs = 15
    for epoch in range(epochs):
        running_loss = 0.0
        correct_total = 0  # On initialise les compteurs pour l'accuracy
        samples_total = 0
        
        # On ajoute une barre de progression avec tqdm
        loop = tqdm.tqdm(train_loader, leave=True)
        for batch_idx, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # --- CALCUL DE L'ACCURACY SUR LE BATCH ---
            # On récupère l'indice du score maximum (ex: indice 3 si c'est un "3")
            _, predicted = torch.max(outputs.data, 1)
            
            # On met à jour les compteurs globaux de l'époque
            samples_total += labels.size(0)
            correct_total += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            
            # Mise à jour du message dans la barre tqdm
            if batch_idx % 10 == 0:
                acc_actuelle = 100 * correct_total / samples_total
                loop.set_description(f"Époque [{epoch+1}/{epochs}]")
                loop.set_postfix(loss=loss.item(), acc=f"{acc_actuelle:.2f}%")

        # --- À LA FIN DE CHAQUE ÉPOQUE ---
        accuracy_finale = 100 * correct_total / samples_total
        print(f"\n✅ Époque {epoch+1} terminée ! Accuracy: {accuracy_finale:.2f}% | Loss moyenne: {running_loss/len(train_loader):.4f}")

    print("\n" + "="*30)
    print(f"SCORE FINAL : {accuracy_finale:.2f}% de réussite")
    print("="*30)

    print("\nEntraînement terminé !")
    
    # 7. Sauvegarde du modèle
    torch.save(model.state_dict(), "mnist_model.pth")
    
    print("Modèle sauvegardé !")

    predict_kaggle(model, device)


def predict_kaggle(model, device):
    print("\n--- Génération des prédictions pour Kaggle ---")
    
    # 1. Charger le fichier de test (attention, pas de labels ici !)
    # On réutilise ta classe MNISTDataset avec is_test=True
    test_dataset = MNISTDataset('./data/test.csv', is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model.eval() # Mode examen
    all_preds = []
    
    with torch.no_grad(): # Pas de calcul de gradients
        for images in tqdm.tqdm(test_loader):
            images = images.to(device)
            
            # Passer les images dans l'IA
            outputs = model(images)
            
            # Outputs contient 10 scores. On prend l'indice du score MAX.
            # ex: si l'indice 4 a le plus gros score, l'IA dit que c'est un "4"
            _, predictions = torch.max(outputs, 1)
            
            all_preds.extend(predictions.cpu().numpy())
    
    # 2. Créer le fichier de soumission
    # Kaggle veut deux colonnes : ImageId (de 1 à 28000) et Label
    submission = pd.DataFrame({
        "ImageId": range(1, len(all_preds) + 1),
        "Label": all_preds
    })
    
    submission.to_csv("submission.csv", index=False)
    print("Fichier 'submission.csv' créé ! Tu peux l'uploader sur Kaggle.")

if __name__ == "__main__":
    main()