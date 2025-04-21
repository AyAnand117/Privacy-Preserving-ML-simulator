# Utility Funcitons
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class CreditCardDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        df = df.drop(columns=['Time'])
        self.y = torch.tensor(df['Class'].values, dtype=torch.long)
        X = df.drop(columns=['Class']).values
        self.X = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# Federated Learning Helpers

def load_and_preprocess(df):
    X = df.drop(columns=['Class','Time'], errors='ignore')
    y = df['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.to_numpy()

def train_local_model(X,y):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X,y)
    return model.coef_, model.intercept_

def add_differential_privacy(coef,intercept, epsilon):
    noise_scale = 1.0/epsilon
    noisy_coef = coef + np.random.normal(0,noise_scale,coef.shape)
    noisy_intercept = intercept + np.random.normal(0, noise_scale, intercept.shape)
    return noisy_coef, noisy_intercept

def federated_averaging(coefs, intercepts):
    avg_coef = np.mean(coefs, axis=0)
    avg_intercept = np.mean(intercepts, axis=0)
    return avg_coef, avg_intercept


#---------- Utility Funcitons---------------


