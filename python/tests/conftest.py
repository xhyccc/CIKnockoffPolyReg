"""conftest.py for Python tests."""
import sys
import os

# Make src importable without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
