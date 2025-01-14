import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader


LABEL_MAP = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "surprise": 4
}


class VideoCNN(nn.Module):
    def __init__(self, num_emotions=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_emotions)
        
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        
        return x

class EmotionFusionNet(nn.Module):
    def __init__(self, num_emotions=5):
        super().__init__()
        
        self.video_conv1 = nn.Conv3d(3, 32, kernel_size=3, padding=1)
        self.video_conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.video_conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.video_bn1 = nn.BatchNorm3d(32)
        self.video_bn2 = nn.BatchNorm3d(64)
        self.video_bn3 = nn.BatchNorm3d(128)
        
        self.video_fc = nn.Linear(128 * 2 * 8 * 8, 256)
        
        self.embedding = nn.Embedding(10000, 128)
        self.text_lstm = nn.LSTM(128, 128, batch_first=True)
        self.text_fc = nn.Linear(128, 256)
        
        self.fusion_fc1 = nn.Linear(512, 256)
        self.fusion_fc2 = nn.Linear(256, 128)
        self.fusion_fc3 = nn.Linear(128, num_emotions)
        
        self.dropout = nn.Dropout(0.4)
    
    def forward(self, video, text):
        batch_size = video.size(0)
        
        video = video.permute(0, 2, 1, 3, 4)
        v = F.relu(self.video_bn1(self.video_conv1(video)))
        v = self.pool(v)
        # print(f"After conv1 shape: {v.shape}")
        
        v = F.relu(self.video_bn2(self.video_conv2(v)))
        v = self.pool(v)
        # print(f"After conv2 shape: {v.shape}")
        
        v = F.relu(self.video_bn3(self.video_conv3(v)))
        v = self.pool(v)
        # print(f"After conv3 shape: {v.shape}")
        
        v = v.view(batch_size, -1)
        # print(f"After flatten shape: {v.shape}")
        
        v = self.video_fc(v)
        # print(f"After video projection shape: {v.shape}")
        
        t = self.embedding(text)
        # print(f"After embedding shape: {t.shape}")
        
        t, (hidden, _) = self.text_lstm(t)
        t = hidden[-1]
        # print(f"After LSTM shape: {t.shape}")
        
        t = self.text_fc(t)
        # print(f"After text projection shape: {t.shape}")
        
        combined = torch.cat((v, t), dim=1)
        # print(f"After concatenation shape: {combined.shape}")
        
        combined = F.relu(self.fusion_fc1(self.dropout(combined)))
        combined = F.relu(self.fusion_fc2(self.dropout(combined)))
        output = self.fusion_fc3(combined)
        
        # print(f"Final output shape: {output.shape}")
        return output


class EmotionDataset(Dataset):
    def __init__(self, video_paths, subtitles, labels, transform=None):
        self.video_paths = video_paths
        self.subtitles = subtitles
        self.labels = labels
        self.transform = transform
        
        self.word2idx = {}
        self.build_vocab()
        
    def build_vocab(self):
        words = set()
        for text in self.subtitles:
            words.update(text.lower().split())
        self.word2idx = {word: idx for idx, word in enumerate(words, 1)}
        
    def process_text(self, text, max_length=50):
        words = text.lower().split()
        indices = [self.word2idx.get(word, 0) for word in words[:max_length]]
        indices = indices + [0] * (max_length - len(indices))
        return torch.tensor(indices)
        
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return torch.zeros((3, 64, 64))

        while len(frames) < 16:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (64, 64))
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            print(f"No frames extracted from video: {video_path}")
            return torch.zeros((3, 64, 64))

        while len(frames) < 16:
            frames.append(torch.zeros_like(frames[0]))

        return torch.stack(frames)

        
    def __len__(self):
        return len(self.video_paths) 

    def __getitem__(self, idx):
        video = self.process_video(self.video_paths[idx])
        text = self.process_text(self.subtitles[idx])
    
        label_str = self.labels[idx]
        label = torch.tensor(LABEL_MAP[label_str], dtype=torch.long)
    
        return video, text, label



def calculate_class_weights(labels):
    """
    Calculate class weights inversely proportional to class frequencies
    """
    label_counts = {}
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    
    total_samples = len(labels)
    class_weights = {}
    for emotion, count in label_counts.items():
        if emotion == "neutral":
            class_weights[LABEL_MAP[emotion]] = 0.501
        if emotion == "joy":
            class_weights[LABEL_MAP[emotion]] = 0.816
        if emotion == "anger":
            class_weights[LABEL_MAP[emotion]] = 0.883
        if emotion == "surprise":
            class_weights[LABEL_MAP[emotion]] = 0.8729
        if emotion == "sadness":
            class_weights[LABEL_MAP[emotion]] = 0.928
        else:
            class_weights[LABEL_MAP[emotion]] = total_samples / (len(label_counts) * count)
    
    weights = torch.FloatTensor([class_weights[i] for i in range(len(LABEL_MAP))])
    return weights


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    early_stopping_patience = 3
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (videos, texts, labels) in enumerate(train_loader):
            videos, texts, labels = videos.to(device), texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if isinstance(model, EmotionFusionNet):
                outputs = model(videos, texts)
            else:
                outputs = model(videos)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for videos, texts, labels in val_loader:
                videos, texts, labels = videos.to(device), texts.to(device), labels.to(device)
                
                if isinstance(model, EmotionFusionNet):
                    outputs = model(videos, texts)
                else:
                    outputs = model(videos)
                    
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.3f} | Val Acc: {val_acc:.2f}%')
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break
