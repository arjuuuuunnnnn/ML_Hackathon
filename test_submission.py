import torch
from model import EmotionFusionNet, EmotionDataset
from torchvision import transforms
import pandas as pd
import os
from torch.utils.data import DataLoader

# Define the label map
LABEL_MAP = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "surprise": 4
}

def predict_test_set(test_csv_path, video_folder_path, model_path='emotion_model.pth', batch_size=8):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Read test CSV
    df = pd.read_csv(test_csv_path, encoding='cp1252')
    
    # Store Sr No. for later use
    sr_numbers = df['Sr No.'].tolist()
    
    # Prepare data paths
    video_paths = [
        os.path.join(video_folder_path, f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4")
        for _, row in df.iterrows()
    ]
    subtitles = df['Utterance'].tolist()
    
    # Create dummy labels (required by dataset class but won't be used)
    dummy_labels = ['neutral'] * len(video_paths)
    
    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create dataset
    test_dataset = EmotionDataset(
        video_paths=video_paths,
        subtitles=subtitles,
        labels=dummy_labels,
        transform=transform
    )
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    model = EmotionFusionNet(num_emotions=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Lists to store predictions
    all_predictions = []
    
    # Predict
    print("Making predictions...")
    with torch.no_grad():
        for batch_idx, (videos, texts, _) in enumerate(test_loader):
            videos = videos.to(device)
            texts = texts.to(device)
            
            outputs = model(videos, texts)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            
            # Convert predictions to emotion labels
            reverse_label_map = {v: k for k, v in LABEL_MAP.items()}
            batch_predictions = [reverse_label_map[pred.item()] for pred in predicted_classes]
            all_predictions.extend(batch_predictions)
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * batch_size} samples...")
    
    # Create submission DataFrame with original Sr No.
    submission_df = pd.DataFrame({
        'Sr No.': sr_numbers,
        'Emotion': all_predictions
    })
    
    return submission_df

def main():
    # Configuration
    test_csv_path = 'data/test.csv'  # Path to your test CSV
    video_folder_path = 'data/test_videos'  # Path to test videos folder
    model_path = 'emotion_model.pth'  # Path to your trained model
    output_path = 'submission.csv'  # Where to save the submission file
    
    try:
        # Make predictions
        print("Starting prediction process...")
        submission_df = predict_test_set(
            test_csv_path=test_csv_path,
            video_folder_path=video_folder_path,
            model_path=model_path
        )
        
        # Save submission file
        submission_df.to_csv(output_path, index=False)
        print(f"Submission file saved to {output_path}")
        
        # Display first few predictions
        print("\nFirst few predictions:")
        print(submission_df.head())
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
