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

# Load metadata (original text data)
metadata = data["metadata"]  # Load metadata as a dictionary list
text_labels = [str(record[selected_key.replace("embeddings_", "")]) for record in metadata]

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

# Filter embeddings and labels for only the top ones
closest_embeddings = embeddings[top_indices]
closest_labels = [text_labels[i] for i in top_indices]

# Define distinct colors for each embedding
colors = sns.color_palette("husl", len(embeddings))

# Create 3D Vector Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot ALL embeddings as arrows (without labels)
for i, vec in enumerate(embeddings):
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
              color=colors[i], linewidth=1, arrow_length_ratio=0.1)

# Plot ONLY the closest embeddings with labels
for i, vec in enumerate(closest_embeddings):
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
              color="blue", linewidth=2, arrow_length_ratio=0.1)  # Highlight closest ones

    # Add text label only for closest embeddings
    ax.text(vec[0], vec[1], vec[2], closest_labels[i], fontsize=8, color="blue")

# Plot the query vector as a red arrow
ax.quiver(0, 0, 0, query_vector[0], query_vector[1], query_vector[2], 
          color="red", linewidth=2, arrow_length_ratio=0.1, label="Query Vector")

# Add a label for the query vector
ax.text(query_vector[0], query_vector[1], query_vector[2], "Query", fontsize=10, color="red", fontweight="bold")

# Draw lines to the top N most similar vectors (highest cosine similarity)
for i in range(len(closest_embeddings)):
    ax.plot([query_vector[0], closest_embeddings[i, 0]],
            [query_vector[1], closest_embeddings[i, 1]],
            [query_vector[2], closest_embeddings[i, 2]], 
            color="black", linestyle="dotted", linewidth=1)

# Set axis labels and title
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title(f"Query + Closest Vectors ({selected_key}) with Cosine Similarity")

# Add legend
ax.legend()

# Show plot
plt.show()
