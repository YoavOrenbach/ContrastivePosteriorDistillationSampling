import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch.optim as optim
from abc import ABC, abstractmethod


def pad_features(features, max_size):
    """Pad features to max_size with zeros"""
    current_size = features.shape[-1]
    if current_size > max_size:
        return features[..., :max_size]
    elif current_size < max_size:
        padding_size = max_size - current_size
        return torch.nn.functional.pad(features, (0, padding_size), mode='constant', value=0)
    return features


class PatchSelector (ABC):
    """Base abstract class for patch selection strategies"""

    @abstractmethod
    def select_patches(self, features, n_patches):
        pass

    @abstractmethod
    def train_step(self, features, labels):
        pass

    @abstractmethod
    def get_accuracy(self, features, labels):
        pass


class RandomPatchSelector(PatchSelector):
    """Random patch selector class using random permutations"""
    def __init__(self):
        super().__init__()

    def select_patches(self, features, n_patches):
        patch_ids = np.random.permutation(len(features))[:n_patches]
        return patch_ids

    def train_step(self, features, labels):
        return 0.0

    def get_accuracy(self, features, labels):
        random_labels = np.random.randint(0, 2, size=len(labels))
        accuracy = 100 * accuracy_score(random_labels, labels)
        return accuracy


class MetaLearnerPatchSelector(PatchSelector):
    """Meta-Learner patch selector class using a neural-network to predict patches"""
    def __init__(self, device, input_dim, hidden_dim=64):
        super().__init__()
        self.feature_dim = input_dim
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        ).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)

    def select_patches(self, features, n_patches):
        with torch.no_grad():
            flat_features = features.view(-1, features.shape[-1])
            padded_features = pad_features(flat_features, self.feature_dim)
            importance_scores = self.network(padded_features.to(torch.float32))
            _, indices = torch.topk(importance_scores.squeeze(), k=n_patches)
        return indices.cpu().numpy()

    def train_step(self, features, labels):
        self.network.train()
        features, labels = features.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        importance_scores = self.network(features.to(torch.float32))
        loss = nn.BCELoss()(importance_scores, labels.unsqueeze(-1))
        loss.requires_grad = True
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_accuracy(self, features, labels):
        self.network.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            features, labels = features.to(self.device), labels.to(self.device)
            importance_scores = self.network(features.to(torch.float32))
            _, predicted = torch.max(importance_scores, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        return accuracy


class MLPatchSelector(PatchSelector):
    """Machine-Learning patch selector class using an ML classifier to predict patches"""
    def __init__(self, ml_classifier, feature_dim):
        super().__init__()
        self.ml_classifier = ml_classifier
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.is_fitted = False

    def select_patches(self, features, n_patches):
        if not self.is_fitted:
            return np.random.permutation(len(features))[:n_patches]

        flat_features = features.view(-1, features.shape[-1])
        padded_features = pad_features(flat_features, self.feature_dim)
        features_np = padded_features.cpu().numpy()
        features_scaled = self.scaler.transform(features_np)
        probs = self.ml_classifier.predict_proba(features_scaled)[:, 1]
        return np.argsort(probs)[-n_patches:]

    def train_step(self, features, labels):
        features_np = features.cpu().numpy()
        self.scaler.fit(features_np)
        features_scaled = self.scaler.transform(features_np)
        self.ml_classifier.fit(features_scaled, labels.cpu().numpy())
        self.is_fitted = True
        return 0.0

    def get_accuracy(self, features, labels):
        features_np = features.cpu().numpy()
        features_scaled = self.scaler.transform(features_np)
        predicted = self.ml_classifier.predict(features_scaled)
        accuracy = 100 * accuracy_score(predicted, labels)
        return accuracy


class SVMPatchSelector(MLPatchSelector):
    def __init__(self, feature_dim):
        super().__init__(SVC(kernel='rbf', probability=True), feature_dim)


class KNNPatchSelector(MLPatchSelector):
    def __init__(self, feature_dim):
        super().__init__(KNeighborsClassifier(n_neighbors=5), feature_dim)


class RandomForestPatchSelector(MLPatchSelector):
    def __init__(self, feature_dim):
        super().__init__(RandomForestClassifier(n_estimators=100), feature_dim)


class PatchImportanceTracker:
    """Tracks and analyzes patch importance during training"""

    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.patch_features = []
        self.importance_scores = []

    def update(self, features, loss_contribution):
        """Update patch history with new patches and their loss contribution"""

        flat_features = features.view(-1, features.shape[-1])
        padded_features = pad_features(flat_features, self.feature_dim).squeeze()

        self.patch_features.append(padded_features.detach().cpu())
        self.importance_scores.append(loss_contribution.squeeze().detach().cpu())

    def get_training_data(self, top_k_percent=20):
        """Generate training data for patch selectors"""
        all_features = torch.cat(self.patch_features, dim=0)
        all_scores = torch.cat(self.importance_scores, dim=0)

        # Convert to binary labels based on top_k_percent
        threshold = torch.quantile(all_scores.float(), 1 - (top_k_percent / 100))
        labels = (all_scores >= threshold).float()

        self.patch_features.clear()
        self.importance_scores.clear()

        return all_features, labels
