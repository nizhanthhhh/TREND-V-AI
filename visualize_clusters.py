import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("VISUALIZING FASHION CLUSTERS")
print("="*50)

# Load data
try:
    df = pd.read_hdf('data/latent_spaces.h5')
    print("✅ Loaded HDF5 file")
except:
    try:
        df = pd.read_pickle('data/latent_spaces_clustered.pkl')
        print("✅ Loaded pickle file")
    except:
        print("❌ Could not load data file!")
        exit()

if 'cluster' not in df.columns:
    print("❌ No cluster column found. Run latent_space_clustering.py first!")
    exit()

# Prepare data
latent_space = np.stack(df['latent_space'].values)
clusters = df['cluster'].values

print(f"\n📊 Data shape: {latent_space.shape}")
print(f"📊 Number of clusters: {len(np.unique(clusters))}")

# Check for zero vectors
zero_vectors = np.sum(np.sum(latent_space == 0, axis=1) == latent_space.shape[1])
if zero_vectors > 0:
    print(f"⚠️  {zero_vectors} images have zero features (likely failed to load)")
    # Remove zero vectors for better visualization
    valid_idx = np.sum(latent_space == 0, axis=1) < latent_space.shape[1]
    latent_space = latent_space[valid_idx]
    clusters = clusters[valid_idx]
    print(f"   Using {len(latent_space)} valid images for visualization")

# Sample if too many points for t-SNE
if len(latent_space) > 500:
    print(f"\n[1] Sampling 500 images for visualization (out of {len(latent_space)})...")
    np.random.seed(42)
    sample_idx = np.random.choice(len(latent_space), 500, replace=False)
    latent_sample = latent_space[sample_idx]
    clusters_sample = clusters[sample_idx]
else:
    latent_sample = latent_space
    clusters_sample = clusters

# Reduce dimensionality with t-SNE
print("\n[2] Reducing dimensions with t-SNE (this may take a moment)...")
try:
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_sample)-1))
    latent_2d = tsne.fit_transform(latent_sample)
    print("   ✅ t-SNE completed")
except Exception as e:
    print(f"   ❌ t-SNE failed: {e}")
    # Fallback to PCA
    from sklearn.decomposition import PCA
    print("   Trying PCA instead...")
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_sample)
    print("   ✅ PCA completed")

# Create visualization
print("\n[3] Creating visualization...")
fig, ax = plt.subplots(figsize=(14, 10))

# Scatter plot
scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                     c=clusters_sample, cmap='tab10', alpha=0.7, s=50)

# Add cluster centers
for cluster in np.unique(clusters_sample):
    cluster_points = latent_2d[clusters_sample == cluster]
    if len(cluster_points) > 0:
        center = cluster_points.mean(axis=0)
        ax.annotate(f'Cluster {cluster}', center, fontsize=12, 
                   fontweight='bold', color='darkred',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

ax.set_title('Fashion Image Clusters Visualization', fontsize=16, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Cluster ID', fontsize=12)

plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
output_file = 'cluster_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"   ✅ Saved to: {output_file}")

# Also save cluster statistics
print("\n[4] Cluster Statistics:")
cluster_stats = pd.Series(clusters).value_counts().sort_index()
for cluster, count in cluster_stats.items():
    percentage = count/len(clusters)*100
    print(f"   Cluster {cluster}: {count} images ({percentage:.1f}%)")

print("\n" + "="*50)
print("VISUALIZATION COMPLETE!")
print("="*50)
print(f"📁 Output file: {output_file}")
print(f"📊 Total images visualized: {len(latent_sample)}")
print("="*50)