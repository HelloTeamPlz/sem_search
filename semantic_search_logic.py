import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal, QObject


from semantic_search import SemanticSearch

class SemanticSearchLogic:
    def __init__(self, ui_instance):
        """Initializes logic handler for Semantic Search.

        Args:
            ui_instance (SemanticSearchApp): Reference to the UI class instance.
        """
        self.ui = ui_instance
        self.search_engine = SemanticSearch()
        self.selected_npz_file = None

    def update_selected_npz(self):
        """Updates selected NPZ file and populates available columns for search."""
        selected_file = self.ui.npz_dropdown.currentText()

        if selected_file and selected_file != "No NPZ files found":
            self.selected_npz_file = selected_file
            _, _, columns = self.search_engine.load_embeddings(self.selected_npz_file)

            if columns:
                self.ui.update_table_headers(columns)
                self.ui.column_dropdown.clear()
                self.ui.column_dropdown.addItems(columns)
            else:
                QMessageBox.warning(self.ui, "Error", "No columns found in NPZ file.")
        else:
            self.selected_npz_file = None

    def upload_file(self):
        """
        Uploads a CSV file, merges it with the currently selected NPZ file, and updates the NPZ 
        with unique rows and regenerated embeddings in a background thread with progress bar.

        Steps:
        - Opens a file dialog for CSV selection
        - Loads the selected CSV into a DataFrame
        - Merges it with existing NPZ metadata
        - Drops duplicates across all rows
        - Starts a background thread to regenerate embeddings and update the NPZ
        """
        file_dialog = QFileDialog(self.ui)
        file_path, _ = file_dialog.getOpenFileName(self.ui, "Select a CSV File", "", "CSV Files (*.csv)")
        if not file_path:
            return

        new_data = pd.read_csv(file_path, sep=None, engine="python", on_bad_lines="warn")
        new_data.fillna("", inplace=True)
        if new_data.empty:
            QMessageBox.warning(self.ui, "Error", "The selected CSV is empty.")
            return

        if not self.selected_npz_file:
            QMessageBox.warning(self.ui, "Error", "No NPZ file selected to append to.")
            return

        _, existing_metadata, _ = self.search_engine.load_embeddings(self.selected_npz_file)
        combined = pd.concat([existing_metadata, new_data], ignore_index=True)
        combined = combined.drop_duplicates()

        if combined.equals(existing_metadata):
            QMessageBox.information(self.ui, "No Changes", "No new rows to add. NPZ file is up to date.")
            return

        self.ui.progress_bar.setVisible(True)
        self.ui.progress_bar.setValue(0)

        self.worker = EmbeddingWorker(combined, self.selected_npz_file, self.search_engine)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.worker.progress.connect(self.ui.progress_bar.setValue)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(lambda: QMessageBox.information(self.ui, "Upload Successful", "CSV data merged and NPZ file updated."))
        self.worker.finished.connect(lambda: self.ui.progress_bar.setVisible(False))
        self.worker.finished.connect(self.ui.refresh_npz_dropdown)

        self.worker.error.connect(lambda msg: QMessageBox.critical(self.ui, "Error", msg))
        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(lambda: self.ui.progress_bar.setVisible(False))

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def perform_search(self):
        """Executes semantic search based on selected column and query."""
        if not self.selected_npz_file:
            QMessageBox.warning(self.ui, "Error", "Select a valid NPZ file.")
            return

        query = self.ui.query_input.text().strip()
        if not query:
            QMessageBox.warning(self.ui, "Error", "Query cannot be empty.")
            return

        selected_column = self.ui.column_dropdown.currentText()
        embeddings, metadata = self.search_engine.load_embeddings(self.selected_npz_file, selected_column)

        if embeddings is None:
            QMessageBox.warning(self.ui, "Error", f"No embeddings for '{selected_column}'.")
            return

        results = self.search_engine.semantic_search(query, embeddings, metadata)
        columns = list(metadata.columns) + ["Similarity"]  # Ensure correct column name will be reffed in SemanticSearch > semantic_search


        self.ui.update_table_headers(columns)
        self.ui.update_table(results, columns)

    def save_results_to_npz(self):
        """Opens a dialog for selecting CSV files and saving embeddings."""
        dialog = SaveNPZDialog(self)
        dialog.exec()

class SaveNPZDialog(QDialog):
    def __init__(self, parent_logic):
        """Dialog for saving embeddings from CSV files to an NPZ file.

        Args:
            parent_logic (SemanticSearchLogic): Reference to parent logic class.
        """
        super().__init__()
        self.logic = parent_logic
        self.setWindowTitle("Save Embeddings to NPZ")
        self.setGeometry(200, 200, 400, 200)

        layout = QVBoxLayout()
        self.npz_label = QLabel("Enter NPZ File Name:")
        layout.addWidget(self.npz_label)

        self.npz_input = QLineEdit(self)
        layout.addWidget(self.npz_input)

        self.select_csv_button = QPushButton("Select CSV Files")
        self.select_csv_button.clicked.connect(self.select_csv_files)
        layout.addWidget(self.select_csv_button)

        self.save_button = QPushButton("Save NPZ")
        self.save_button.clicked.connect(self.save_npz)
        layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.selected_csv_files = []

    def select_csv_files(self):
        """Selects multiple CSV files for embedding generation."""
        files, _ = QFileDialog.getOpenFileNames(self, "Select CSV Files", "", "CSV Files (*.csv)")
        if files:
            self.selected_csv_files = files
            QMessageBox.information(self, "Files Selected", f"{len(files)} files selected.")

    def save_npz(self):
        """Saves selected CSV files' embeddings to an NPZ file."""
        npz_file_name = self.npz_input.text().strip()
        if not npz_file_name.endswith(".npz"):
            npz_file_name += ".npz"

        if not self.selected_csv_files:
            QMessageBox.warning(self, "Error", "No CSV files selected.")
            return

        data, _ = self.logic.search_engine.load_csv_files(self.selected_csv_files)
        if data.empty:
            QMessageBox.warning(self, "Error", "No valid data in selected CSV files.")
            return

        self.logic.search_engine.save_column_embeddings_to_npz(data, npz_file_name)
        QMessageBox.information(self, "Save Successful", f"Embeddings saved to '{npz_file_name}'.")
        self.logic.ui.refresh_npz_dropdown()
        self.close()

class EmbeddingWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, data, npz_filename, search_engine):
        super().__init__()
        self.data = data
        self.npz_filename = npz_filename
        self.search_engine = search_engine

    def run(self):
        try:
            metadata = self.data.copy()
            embeddings_dict = {}
            total = len(metadata.columns)

            for idx, column in enumerate(metadata.columns):
                sanitized_col = column.replace(" ", "_")
                embeddings = self.search_engine.model.encode(
                    metadata[column].astype(str).tolist(),
                    convert_to_tensor=False
                )
                embeddings_dict[f"embeddings_{sanitized_col}"] = embeddings
                self.progress.emit(int((idx + 1) / total * 100))

            np.savez_compressed(self.npz_filename, metadata=metadata.to_dict(orient='records'), **embeddings_dict)
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))