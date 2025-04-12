import os
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QWidget, QHeaderView, QComboBox,
    QMessageBox, QTableWidgetItem, QProgressBar
)
from semantic_search_logic import SemanticSearchLogic

class SemanticSearchApp(QMainWindow):
    def __init__(self):
        """Initialize the main Semantic Search UI window."""
        super().__init__()

        self.setWindowTitle("Semantic Search App")
        self.setGeometry(100, 100, 900, 600)

        icon_path = os.path.abspath("app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.logic = SemanticSearchLogic(self)

        # Load default theme explicitly
        self.load_stylesheet('styles/light.qss')

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        upload_action = QAction("Upload CSV File", self)
        upload_action.triggered.connect(self.logic.upload_file)
        file_menu.addAction(upload_action)

        save_action = QAction("Save Results to NPZ", self)
        save_action.triggered.connect(self.logic.save_results_to_npz)
        file_menu.addAction(save_action)

        view_menu = menu_bar.addMenu("View")

        dark_style_action = QAction("Dark Theme", self)
        dark_style_action.triggered.connect(lambda: self.load_stylesheet('styles/dark.qss'))
        view_menu.addAction(dark_style_action)

        light_style_action = QAction("Light Theme", self)
        light_style_action.triggered.connect(lambda: self.load_stylesheet('styles/light.qss'))
        view_menu.addAction(light_style_action)

        # Setup central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # NPZ dropdown
        self.npz_label = QLabel("Select NPZ File:")
        layout.addWidget(self.npz_label)

        self.npz_dropdown = QComboBox(self)
        layout.addWidget(self.npz_dropdown)
        self.npz_dropdown.currentIndexChanged.connect(self.logic.update_selected_npz)

        # Column dropdown
        self.column_label = QLabel("Select Column:")
        layout.addWidget(self.column_label)

        self.column_dropdown = QComboBox(self)
        layout.addWidget(self.column_dropdown)

        # Query input
        self.query_label = QLabel("Enter your search query:")
        layout.addWidget(self.query_label)

        self.query_input = QLineEdit(self)
        self.query_input.setPlaceholderText("Type your query here...")
        self.query_input.returnPressed.connect(self.logic.perform_search)
        layout.addWidget(self.query_input)

        # Run button
        self.run_button = QPushButton("🔍 Search")
        self.run_button.clicked.connect(self.logic.perform_search)
        layout.addWidget(self.run_button)

        # Results table
        self.result_table = QTableWidget(self)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.result_table)

        # call refresh_npz_dropdown after UI elements are set up
        self.refresh_npz_dropdown()

        # progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)


    def refresh_npz_dropdown(self):
        """Refreshes the dropdown menu to list available NPZ files."""
        self.npz_dropdown.clear()
        npz_files = [f for f in os.listdir() if f.endswith(".npz")]

        if npz_files:
            self.npz_dropdown.addItems(npz_files)
            self.logic.selected_npz_file = npz_files[0]
            self.npz_dropdown.setCurrentText(npz_files[0])

            # Explicitly call update_selected_npz to load columns
            self.logic.update_selected_npz()
        else:
            self.npz_dropdown.addItem("No NPZ files found")
            self.logic.selected_npz_file = None


    def update_table_headers(self, columns):
        """Updates table headers dynamically.

        Args:
            columns (list[str]): Column names for the table headers.
        """
        self.result_table.setColumnCount(len(columns))
        self.result_table.setHorizontalHeaderLabels([str(col) for col in columns])

    def update_table(self, results, columns):
        """Populates the table with search results.

        Args:
            results (pd.DataFrame): Search results data.
            columns (list[str]): Column names to display.
        """
        self.result_table.setRowCount(0)

        if results.empty:
            QMessageBox.information(self, "No Results", "No matching results found.")
            return

        self.result_table.setRowCount(len(results))

        for row_idx, (_, result) in enumerate(results.iterrows()):
            for col_idx, col_name in enumerate(columns):
                if col_name == "Similarity":
                    value = f"{float(result[col_name]):.2f}" 
                else:
                    value = str(result[col_name])
                self.result_table.setItem(row_idx, col_idx, QTableWidgetItem(value))

    def load_stylesheet(self, stylesheet_name):
        """Loads stylesheet from the styles directory.

        Args:
            stylesheet_name (str): Stylesheet file name.
        """
        try:
            with open(stylesheet_name, 'r') as file:
                self.setStyleSheet(file.read())
        except FileNotFoundError:
            print(f"Stylesheet {stylesheet_name} not found.")
