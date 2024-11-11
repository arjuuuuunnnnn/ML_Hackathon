# Example data preparation
video_paths = []
subtitles = []
labels = []

# Read from your CSV
df = pd.read_csv('train.csv')
for _, row in df.iterrows():
    video_path = f"./S{row['Season']}E{row['Episode']}.mp4"
    video_paths.append(video_path)
    subtitles.append(row['Utterance'])
    labels.append(emotion_to_idx[row['Emotion']])
