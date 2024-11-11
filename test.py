import torch
from model import EmotionFusionNet, EmotionDataset
from torchvision import transforms
import cv2
import os

# Define the label map (same as in main.py)
LABEL_MAP = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "surprise": 4
}

def test_single_video(video_path, text, model_path='emotion_model.pth'):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = EmotionFusionNet(num_emotions=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create a minimal dataset with just one sample
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create dataset with single sample
    dataset = EmotionDataset(
        video_paths=[video_path],
        subtitles=[text],
        labels=['neutral'],  # dummy label
        transform=transform
    )
    
    # Get the single sample
    video, text_tensor, _ = dataset[0]
    
    # Add batch dimension
    video = video.unsqueeze(0).to(device)
    text_tensor = text_tensor.unsqueeze(0).to(device)
    
    # Print shapes for debugging
    print(f"Video shape: {video.shape}")
    print(f"Text shape: {text_tensor.shape}")
    
    # Get prediction
    try:
        with torch.no_grad():
            outputs = model(video, text_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Convert prediction to emotion label using the global LABEL_MAP
        reverse_label_map = {v: k for k, v in LABEL_MAP.items()}
        predicted_emotion = reverse_label_map[predicted_class.item()]
        
        # Get probability distribution
        probs_dict = {reverse_label_map[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        
        return predicted_emotion, probs_dict
    
    except Exception as e:
        print(f"Error during model prediction: {str(e)}")
        raise

def main():
    # Example usage
    video_path = "data/train_videos/dia126_utt5.mp4"  # Replace with your video path
    text = "Stop it! I will kill you. I hate the fact that my roomis very small"  # Replace with your text
    
    try:
        predicted_emotion, probabilities = test_single_video(video_path, text)
        
        print("\nPrediction Results:")
        print(f"Predicted Emotion: {predicted_emotion}")
        print("\nProbabilities for each emotion:")
        for emotion, prob in probabilities.items():
            print(f"{emotion}: {prob:.4f}")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
