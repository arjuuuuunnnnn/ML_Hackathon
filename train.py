from model import EmotionFusionNet, EmotionDataset, train_model, calculate_class_weights
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os

def main():
    df = pd.read_csv('data/train.csv', encoding='cp1252')

    
    video_paths = [
        os.path.join('data', 'train_videos', f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4")
        for _, row in df.iterrows()
    ]

    subtitles = df['Utterance'].tolist()
    labels = df['Emotion'].tolist()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    ])
    
    dataset = EmotionDataset(
        video_paths=video_paths,
        subtitles=subtitles,
        labels=labels,
        transform=transform
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EmotionFusionNet(num_emotions=5)
    model = model.to(device)
    
    train_labels = [labels[i] for i in train_dataset.indices]
    weights = calculate_class_weights(train_labels)
    weights = weights.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,
        device=device
    )
    
    torch.save(model.state_dict(), 'emotion_model.pth')
    print("Training completed. Model saved as 'emotion_model.pth'")

if __name__ == "__main__":
    main()
