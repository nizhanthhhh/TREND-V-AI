import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
from torchvision import models, transforms
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("LATENT SPACE CREATION (PyTorch Version)")
print("="*50)

# Load data
print("\n[1] Loading posts data...")
df = pd.read_csv('data/posts.csv')
print(f"    Loaded {len(df)} posts")

# Check images
print("\n[2] Checking images folder...")
if os.path.exists('images'):
    images = os.listdir('images')
    print(f"    Found {len(images)} images in folder")
else:
    print("    ❌ 'images' folder not found!")
    print("    Please run: python download_images.py")
    exit()

# Load model
print("\n[3] Loading ResNet50 model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    Using device: {device}")

try:
    model = models.resnet50(pretrained=True)
    # Remove the final classification layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    print("    ✅ Model loaded successfully")
except Exception as e:
    print(f"    ❌ Error loading model: {e}")
    exit()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Extract features
print("\n[4] Extracting features from images...")
latent_spaces = []
success_count = 0
fail_count = 0

for idx, row in df.iterrows():
    try:
        # Find image file
        image_path = None
        image_id = row['id']
        
        # Check for common image extensions
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            test_path = f"images/{image_id}{ext}"
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        # If not found by ID, try by index
        if image_path is None:
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                test_path = f"images/{idx+1}{ext}"
                if os.path.exists(test_path):
                    image_path = test_path
                    break
        
        if image_path is None:
            latent_spaces.append(np.zeros(2048))
            fail_count += 1
            continue
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(image_tensor)
            features = features.squeeze().cpu().numpy()
        
        latent_spaces.append(features)
        success_count += 1
        
        # Progress update
        if (idx + 1) % 50 == 0:
            print(f"    Processed {idx + 1}/{len(df)} images")
            
    except Exception as e:
        print(f"    ⚠️  Error on image {row['id']}: {e}")
        latent_spaces.append(np.zeros(2048))
        fail_count += 1

# Add latent space to dataframe
df['latent_space'] = latent_spaces

# Save
print("\n[5] Saving latent space...")
try:
    df.to_hdf('data/latent_spaces.h5', key='df_items', mode='w')
    print("    ✅ Saved to: data/latent_spaces.h5")
except Exception as e:
    print(f"    ❌ Error saving HDF5: {e}")
    # Fallback: save as pickle
    df.to_pickle('data/latent_spaces.pkl')
    print("    ✅ Saved as fallback: data/latent_spaces.pkl")

# Summary
print("\n" + "="*50)
print("COMPLETE!")
print("="*50)
print(f"✅ Successfully processed: {success_count} images")
print(f"⚠️  Failed/Skipped: {fail_count} images")
print(f"📊 Feature vector size: {len(latent_spaces[0]) if latent_spaces else 0}")
print("="*50)