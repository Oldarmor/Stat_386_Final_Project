from pathlib import Path
from setuptools import setup

# Provide minimal metadata (name + version) and long_description so
# legacy setup.py generation will embed the README into PKG-INFO/METADATA.
setup(
    name="stat-386-final-project",
    version="0.1.2",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
)
