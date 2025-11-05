#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
SETUP_FILE="setup.py"

# --- Script Logic ---

# 1. Check for new version argument
if [ -z "$1" ]; then
  echo "Error: No version number supplied."
  echo "Usage: ./release.sh <new-version>"
  exit 1
fi

NEW_VERSION=$1
echo "🚀 Starting release process for version $NEW_VERSION..."

# 2. Update version in setup.py
echo "🔍 Finding current version in $SETUP_FILE..."
CURRENT_VERSION=$(grep "version=" $SETUP_FILE | sed -E 's/.*version="([^"]+)".*/\1/')
if [ -z "$CURRENT_VERSION" ]; then
    echo "❌ Error: Could not find the version string in $SETUP_FILE."
    exit 1
fi
echo "Found current version: $CURRENT_VERSION"

echo "🔄 Updating version in $SETUP_FILE to $NEW_VERSION..."
sed -i "s/version=\"$CURRENT_VERSION\"/version=\"$NEW_VERSION\"/" $SETUP_FILE
echo "✅ Version updated." 
# 3. Clean up old builds
echo "🧹 Cleaning up old build artifacts..."
rm -rf build/ dist/ *.egg-info/
echo "✅ Cleanup complete."

# 4. Build the package
echo "📦 Building source and wheel distributions..."
python3 setup.py sdist bdist_wheel
echo "✅ Build successful. New packages are in dist/:"
ls -l dist

# 5. Upload to PyPI
echo "☁️  Uploading to PyPI..."
echo "Make sure you have 'twine' installed (pip install twine) and your ~/.pypirc is configured."
twine upload dist/*

echo "🎉 Successfully published version $NEW_VERSION to PyPI!"
