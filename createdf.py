import pandas as pd
import os

def create_df(data_dir):
    # List all files in the directory
    files = os.listdir(data_dir)
    
    # Initialize lists to store image paths and labels
    image_paths = []
    labels = []
    
    for file in files:
        # Assuming files are in the format 'label_imageid.jpg'
        label = file.split('_')[0]  # Extract label from filename
        image_path = os.path.join(data_dir, file)  # Full path to image
        image_paths.append(image_path)
        labels.append(label)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })
    
    return df

# Use the function to create DataFrames
train_df = create_df("config/train")
val_df = create_df("config/valid")
test_df = create_df("config/test")
print(train_df)
print(val_df)
