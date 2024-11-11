from model import EmotionFusionNet, EmotionDataset, train_model, calculate_class_weights
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os

def main():
    # 1. Read the CSV file
    df = pd.read_csv('data/train.csv', encoding='cp1252')
    
    # 2. Prepare data paths - assuming train.csv has video_name, text, and label columns
    # video_paths = [os.path.join('data/train_videos', name) for name in df['video_name']]
    
    video_paths = [
        os.path.join('data', 'train_videos', f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4")
        for _, row in df.iterrows()
    ]



    subtitles = df['Utterance'].tolist()
    labels = df['Emotion'].tolist()
    
    # 3. Create transform for videos
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # 4. Create dataset
    dataset = EmotionDataset(
        video_paths=video_paths,
        subtitles=subtitles,
        labels=labels,
        transform=transform
    )
    
    # 5. Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 6. Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 7. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 8. Create model
    model = EmotionFusionNet(num_emotions=5)  # adjust num_emotions based on your classes
    model = model.to(device)
    
    # 8.5 Calculate class weights
    train_labels = [labels[i] for i in train_dataset.indices]
    weights = calculate_class_weights(train_labels)
    weights = weights.to(device)

    # 9. Setup loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 10. Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device
    )
    
    # 11. Save the model
    torch.save(model.state_dict(), 'emotion_model.pth')
    print("Training completed. Model saved as 'emotion_model.pth'")

if __name__ == "__main__":
    main()
