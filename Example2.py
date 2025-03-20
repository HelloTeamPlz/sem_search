import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

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

# Load Sentence Transformer model for query encoding
model = SentenceTransformer("all-MiniLM-L6-v2")

# User inputs a query
query_text = input("Enter your search query: ")  # User provides a query
query_vector = model.encode([query_text])[0]  # Convert query to an embedding

# Ensure query_vector is also 3D if embeddings are
if query_vector.shape[0] > 3:
    query_vector = query_vector[:3]

# Compute cosine similarity between the query and all embeddings
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = np.array([cosine_similarity(query_vector, emb) for emb in embeddings])

# Get indices of the 5 most similar vectors
top_n = 5
top_indices = np.argsort(similarities)[-top_n:]  # Sort by highest similarity

# Define distinct colors for each embedding
colors = sns.color_palette("husl", len(embeddings))

# Create 3D Scatter Plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot all embeddings as dots
ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], 
           c=colors, s=50, alpha=0.8, edgecolors=None, marker='o', label="Embeddings")

# Plot the query vector as a red dot
ax.scatter(query_vector[0], query_vector[1], query_vector[2], 
           c="red", s=100, edgecolors="black", marker="o", label="Query Vector")

# Draw lines to the top N most similar vectors (highest cosine similarity)
for i in top_indices:
    ax.plot([query_vector[0], embeddings[i, 0]],
            [query_vector[1], embeddings[i, 1]],
            [query_vector[2], embeddings[i, 2]], 
            color="black", linestyle="dotted", linewidth=1)

# Set axis labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title(f"3D Embedding Scatter Plot ({selected_key}) with Cosine Similarity")

# Add legend
ax.legend()

# Show plot
plt.show()
