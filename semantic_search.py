import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initializes SemanticSearch with a sentence transformer model.

        Args:
            model_name (str, optional): Name of the sentence transformer model. Defaults to "all-MiniLM-L6-v2".
        """
        self.model = SentenceTransformer(model_name)

    def load_csv_files(self, file_paths):
        """Loads and combines data from multiple CSV files.

        Args:
            file_paths (list[str]): Paths to CSV files to load.

        Returns:
            tuple: Combined DataFrame and list of common columns.
        """
        combined_data = []
        common_columns = None

        for file_path in file_paths:
            df = pd.read_csv(file_path, sep=None, engine="python", on_bad_lines="warn")
            if common_columns is None:
                common_columns = df.columns.tolist()
            elif set(common_columns) != set(df.columns):
                continue

            df.fillna("", inplace=True)
            combined_data.append(df)

        return pd.concat(combined_data, ignore_index=True), common_columns

    def save_column_embeddings_to_npz(self, data, npz_filename):
        """Saves embeddings for each column to an NPZ file.

        Args:
            data (pd.DataFrame): Data to encode.
            npz_filename (str): Output NPZ file.
        """
        metadata = data.copy()
        embeddings_dict = {}

        for column in data.columns:
            sanitized_col = column.replace(" ", "_")
            embeddings = self.model.encode(data[column].astype(str).tolist(), convert_to_tensor=False)
            embeddings_dict[f"embeddings_{sanitized_col}"] = embeddings

        np.savez_compressed(npz_filename, metadata=metadata.to_dict(orient='records'), **embeddings_dict)

    def load_embeddings(self, npz_filename, column_name=None):
        """Loads embeddings and metadata from an NPZ file.

        Args:
            npz_filename (str): Path to NPZ file.
            column_name (str, optional): Specific column to load embeddings for. Defaults to None.

        Returns:
            tuple: embeddings, metadata DataFrame, and optionally list of columns.
        """
        data = np.load(npz_filename, allow_pickle=True)
        metadata = pd.DataFrame.from_records(data['metadata'].tolist())

        if column_name:
            embeddings_key = f"embeddings_{column_name.replace(' ', '_')}"
            embeddings = data[embeddings_key]
            return embeddings, metadata
        else:
            columns = [key.replace("embeddings_", "") for key in data.files if key.startswith("embeddings_")]
            return None, metadata, columns

    def semantic_search(self, query, embeddings, metadata, top_n=5):
        """Performs semantic search using embeddings.

        Args:
            query (str): Search query.
            embeddings (np.ndarray): Pre-computed embeddings.
            metadata (pd.DataFrame): Associated metadata.
            top_n (int, optional): Number of results. Defaults to 5.

        Returns:
            pd.DataFrame: Top results with similarity.
        """
        query_embedding = self.model.encode([query], convert_to_tensor=False)[0]
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        results_df = metadata.copy()
        results_df["Similarity"] = similarities.round(2)  # final output for sim

        return results_df.sort_values(by="Similarity", ascending=False).head(top_n)

