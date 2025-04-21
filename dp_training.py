
# dp_training.py
# Core pytorch + Opacus Pipieline

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from opacus import PrivacyEngine
from utils import CreditCardDataset
from models import MLP
from sklearn.metrics import classification_report

def train_with_dp(data_path, noise_multiplier=1.0,max_grad_norm=1.0,epochs=5, batch_size=256, return_eps_curve=False):
    dataset = CreditCardDataset(data_path)
    train_len = int(0.8 * len(dataset))
    train_data, test_data = random_split(dataset,[train_len, len(dataset)-train_len])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = MLP(input_dim=dataset.X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)     
    criterion = nn.CrossEntropyLoss()

    #DP Integration
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module = model,
        optimizer = optimizer,
        data_loader = train_loader,
        noise_multiplier = noise_multiplier,
        max_grad_norm = max_grad_norm,
        )
     
     
    #Tacking epsilon after each epoch 
    epsilon_history=[]

    # Training

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits,y_batch)
            loss.backward()
            optimizer.step()

    # Get epsilon after each epoch

        epsilon= privacy_engine.get_epsilon(delta=1e-5)
        epsilon_history.append(epsilon)
        print(f"Epoch {epoch+1}/{epochs}- Eps = {epsilon:.3f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [],[]

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = torch.argmax(model(X_batch), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    print("Classification Report (DP Model):") 
    report = classification_report(all_labels, all_preds, zero_division=0)
    print(report)

     # Privacy Budget
    print(f"Privacy Budget: Eps = {epsilon_history[-1]:.2f}, Delta = 1e-5")

    final_eps = epsilon_history[-1]
    if return_eps_curve:
        return report, final_eps, epsilon_history
    else:
        return report, final_eps, None

