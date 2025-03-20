import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the NPZ file
npz_file = "asd.npz"  # Change this to your actual NPZ file path
data = np.load(npz_file, allow_pickle=True)

# Get all keys in the NPZ file
keys = data.files
print("Keys in NPZ file:", keys)

# Find embedding keys dynamically
embedding_keys = [key for key in keys if "embedding" in key.lower()]
if not embedding_keys:
    raise ValueError("No embedding keys found in NPZ file!")

# Select the first embedding key for visualization
selected_key = embedding_keys[0]
embeddings = data[selected_key]  # Load embeddings from the chosen key

# Ensure embeddings are at most 3D for visualization
if embeddings.shape[1] > 3:
    embeddings = embeddings[:, :3]  # Take only the first 3 dimensions

# Define distinct colors for each embedding
colors = sns.color_palette("husl", len(embeddings))

# Create 3D Scatter Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each embedding as a dot
ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], 
           c=colors, s=50, alpha=0.8, edgecolors='k')

# Set axis labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title(f"3D Embedding Scatter Plot ({selected_key})")

# Show plot
plt.show()
