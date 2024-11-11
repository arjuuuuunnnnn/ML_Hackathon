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
        # Input shape: (batch_size, 3, 64, 64)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Calculate final feature map size
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_emotions)
        
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        
        return x

# 2. Custom Fusion Model
# class EmotionFusionNet(nn.Module):
#     def __init__(self, num_emotions=5):
#         super().__init__()
        
#         # Video Processing Branch
#         self.video_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.video_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.video_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.video_bn1 = nn.BatchNorm2d(32)
#         self.video_bn2 = nn.BatchNorm2d(64)
#         self.video_bn3 = nn.BatchNorm2d(128)
        
#         self.video_fc = nn.Linear(128 * 8 * 8, 512)

#         # Text Processing Branch
#         self.embedding = nn.Embedding(10000, 128)  # Vocabulary size of 10000
#         self.text_lstm = nn.LSTM(128, 256, batch_first=True, bidirectional=True)
        
#         self.text_fc = nn.Linear(2 * 256, 128)  # 2 * 256 because LSTM is bidirectional

#         # Fusion Layers
#         self.fusion_fc1 = nn.Linear(128 * 8 * 8 + 512, 512)  # Concatenated features
#         self.fusion_fc2 = nn.Linear(512, 256)
#         self.fusion_fc3 = nn.Linear(256, num_emotions)
        
#         self.dropout = nn.Dropout(0.3) 

#     def forward(self, video, text):
#         # Video Branch
#         batch_size = video.size(0)

#         print(f"Input shapes - Video: {video.shape}, Text: {text.shape}")

#         video = video.reshape(batch_size, 3, 64, 64)

#         v = F.relu(self.video_bn1(self.video_conv1(video)))
#         v = self.pool(v)
#         v = F.relu(self.video_bn2(self.video_conv2(v)))
#         v = self.pool(v)
#         v = F.relu(self.video_bn3(self.video_conv3(v)))
#         v = self.pool(v)

#         print(f"After convolutions shape: {v.shape}")
        
#         v = v.view(batch_size, -1)  # Flatten the video tensor
#         print(f"After flatten shape: {v.shape}")
#         v = self.video_fc(v)
#         print(f"After video projection shape: {v.shape}")

#         # Text Branch
#         t = self.embedding(text)  # Apply embedding
#         print(f"After embedding shape: {t.shape}")
#         t, (hidden, _) = self.text_lstm(t)  # Get the LSTM output
#         t = hidden[-1]

#         print(f"After LSTM shape: {t.shape}")

#         # t = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # Concatenate bidirectional LSTM
#         t = self.text_fc(t)
        
#         print(f"After text projection shape: {t.shape}")
        
#         # Ensure shapes match before concatenation
#         print(f"Shapes before concatenation - Video: {v.shape}, Text: {t.shape}")

#         # Fusion
#         combined = torch.cat((v, t), dim=1)  # Concatenate along dimension 1
#         print(f"After concatenation shape: {combined.shape}")
#         combined = F.relu(self.fusion_fc1(self.dropout(combined)))
#         combined = F.relu(self.fusion_fc2(self.dropout(combined)))
#         output = self.fusion_fc3(combined)
#         print(f"Final output shape: {output.shape}")
#         return output


class EmotionFusionNet(nn.Module):
    def __init__(self, num_emotions=5):
        super().__init__()
        
        # 3D Video Processing Branch (for multiple frames)
        self.video_conv1 = nn.Conv3d(3, 32, kernel_size=3, padding=1)
        self.video_conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.video_conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.video_bn1 = nn.BatchNorm3d(32)
        self.video_bn2 = nn.BatchNorm3d(64)
        self.video_bn3 = nn.BatchNorm3d(128)
        
        # Calculate the size after 3D convolutions
        # Input: [batch, 3, 16, 64, 64]
        # After 3 pools: [batch, 128, 2, 8, 8]
        self.video_fc = nn.Linear(128 * 2 * 8 * 8, 128)
        
        # Text Processing Branch
        self.embedding = nn.Embedding(10000, 64)
        self.text_lstm = nn.LSTM(64, 64, batch_first=True)
        self.text_fc = nn.Linear(64, 128)
        
        # Fusion Layers
        self.fusion_fc1 = nn.Linear(256, 128)
        self.fusion_fc2 = nn.Linear(128, 64)
        self.fusion_fc3 = nn.Linear(64, num_emotions)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, video, text):
        batch_size = video.size(0)
        
        # Debug print for input shapes
        print(f"Input shapes - Video: {video.shape}, Text: {text.shape}")
        
        # Video Branch
        # Permute the dimensions from [batch, frames, channels, height, width] to [batch, channels, frames, height, width]
        video = video.permute(0, 2, 1, 3, 4)
        print(f"After permute shape: {video.shape}")
        
        # 3D CNN layers
        v = F.relu(self.video_bn1(self.video_conv1(video)))
        v = self.pool(v)
        print(f"After conv1 shape: {v.shape}")
        
        v = F.relu(self.video_bn2(self.video_conv2(v)))
        v = self.pool(v)
        print(f"After conv2 shape: {v.shape}")
        
        v = F.relu(self.video_bn3(self.video_conv3(v)))
        v = self.pool(v)
        print(f"After conv3 shape: {v.shape}")
        
        # Flatten
        v = v.view(batch_size, -1)
        print(f"After flatten shape: {v.shape}")
        
        # Project video features
        v = self.video_fc(v)
        print(f"After video projection shape: {v.shape}")
        
        # Text Branch
        t = self.embedding(text)
        print(f"After embedding shape: {t.shape}")
        
        t, (hidden, _) = self.text_lstm(t)
        t = hidden[-1]
        print(f"After LSTM shape: {t.shape}")
        
        t = self.text_fc(t)
        print(f"After text projection shape: {t.shape}")
        
        # Concatenate features
        combined = torch.cat((v, t), dim=1)
        print(f"After concatenation shape: {combined.shape}")
        
        # Fusion layers
        combined = F.relu(self.fusion_fc1(self.dropout(combined)))
        combined = F.relu(self.fusion_fc2(self.dropout(combined)))
        output = self.fusion_fc3(combined)
        
        print(f"Final output shape: {output.shape}")
        return output



# 3. Data Processing
class EmotionDataset(Dataset):
    def __init__(self, video_paths, subtitles, labels, transform=None):
        self.video_paths = video_paths
        self.subtitles = subtitles
        self.labels = labels
        self.transform = transform
        
        # Create vocabulary for text
        self.word2idx = {}
        self.build_vocab()
        
    def build_vocab(self):
        # Build vocabulary from subtitles
        words = set()
        for text in self.subtitles:
            words.update(text.lower().split())
        self.word2idx = {word: idx for idx, word in enumerate(words, 1)}
        
    def process_text(self, text, max_length=50):
        # Convert text to indices
        words = text.lower().split()
        indices = [self.word2idx.get(word, 0) for word in words[:max_length]]
        indices = indices + [0] * (max_length - len(indices))  # Padding
        return torch.tensor(indices)
        
    # def process_video(self, video_path):
    #     # Extract face from video frame
    #     cap = cv2.VideoCapture(video_path)
    #     frames = []
    #     while len(frames) < 16:  # Get 16 frames
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         frame = cv2.resize(frame, (64, 64))
    #         if self.transform:
    #             frame = self.transform(frame)
    #         frames.append(frame)
    #     cap.release()
        
    #     # Pad if necessary
    #     while len(frames) < 16:
    #         frames.append(torch.zeros_like(frames[0]))
            
    #     return torch.stack(frames)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return torch.zeros((3, 64, 64))  # Return a default frame

        while len(frames) < 16:  # Aim to get 16 frames
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (64, 64))
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        # Check if frames are empty
        if len(frames) == 0:
            print(f"No frames extracted from video: {video_path}")
            return torch.zeros((3, 64, 64))

        # Pad with zeros if necessary
        while len(frames) < 16:
            frames.append(torch.zeros_like(frames[0]))

        return torch.stack(frames)

        
    def __len__(self):
        return len(self.video_paths)
        
    # def __getitem__(self, idx):
    #     video = self.process_video(self.video_paths[idx])
    #     text = self.process_text(self.subtitles[idx])
    #     label = self.labels[idx]
    #     return video, text, label

    def __getitem__(self, idx):
        video = self.process_video(self.video_paths[idx])
        text = self.process_text(self.subtitles[idx])
    
        # Map label string to an integer using LABEL_MAP
        label_str = self.labels[idx]
        label = torch.tensor(LABEL_MAP[label_str], dtype=torch.long)
    
        return video, text, label
 

# 4. Training Functions
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
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
        
        # Validation
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