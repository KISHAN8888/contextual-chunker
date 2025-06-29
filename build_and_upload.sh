#!/bin/bash

# Build and upload script for document-chunker package

set -e

echo "Building document-chunker package..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build the package
python -m build

# Check the package
python -m twine check dist/*

echo "Package built successfully!"
echo "To upload to PyPI:"
echo "1. Test PyPI: python -m twine upload --repository testpypi dist/*"
echo "2. Real PyPI: python -m twine upload dist/*"

echo ""
echo "Files in dist/:"
ls -la dist/