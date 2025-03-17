import sys
from PyQt6.QtWidgets import QApplication
from semantic_search_ui import SemanticSearchApp  # Import the UI class

def main():
    """Launch the PyQt6 Semantic Search UI."""
    app = QApplication(sys.argv)  # Create QApplication instance
    window = SemanticSearchApp()  # Create UI window
    window.show()  # Show the window
    sys.exit(app.exec())  # Start event loop

if __name__ == "__main__":
    main()


