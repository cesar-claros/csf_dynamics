"""Imagenette training-dynamics pilot."""
import os

# The paper classes show a progress bar per sample. Set this before tqdm loads.
os.environ.setdefault("TQDM_DISABLE", "1")
