#!/bin/bash
# Build deployment package for Bedrock AgentCore direct code deployment
# This script creates an ARM64-compatible ZIP package with all dependencies

set -e  # Exit on any error

PYTHON_VERSION="3.11"
PACKAGE_DIR="deployment_package"
OUTPUT_ZIP="deployment_package.zip"

# Source files to include (from agent-container directory)
SOURCE_FILES=(
    "agent-container/server.py"
    "agent-container/rag_agent.py"
    "agent-container/ask_knowledgebase.py"
)

echo "========================================================================"
echo "Building Bedrock AgentCore Deployment Package"
echo "========================================================================"
echo ""
echo "Python Version: ${PYTHON_VERSION}"
echo "Architecture:   ARM64 (linux/aarch64)"
echo "Package Format: ZIP"
echo ""

# Clean previous builds
echo "→ Cleaning previous builds..."
rm -rf ${PACKAGE_DIR} ${OUTPUT_ZIP}

# Install dependencies for ARM64 architecture
echo "→ Installing dependencies for ARM64..."
uv pip install \
    --python-platform aarch64-manylinux2014 \
    --python-version ${PYTHON_VERSION} \
    --target=${PACKAGE_DIR} \
    --only-binary=:all: \
    -r pyproject.toml

if [ $? -ne 0 ]; then
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Create ZIP from dependencies
echo "→ Creating ZIP archive from dependencies..."
cd ${PACKAGE_DIR}
zip -r ../${OUTPUT_ZIP} . -q
cd ..

# Add source files to ZIP
echo "→ Adding source files to ZIP..."
for file in "${SOURCE_FILES[@]}"; do
    if [ -f "$file" ]; then
        # Extract just the filename (remove directory path)
        filename=$(basename "$file")
        echo "  - ${filename}"
        # Add to root of zip
        zip ${OUTPUT_ZIP} "$file" -j -q  # -j: junk (don't record) directory names
    else
        echo "✗ Warning: Source file not found: $file"
    fi
done

# Set proper permissions (required by AgentCore)
# Files: 644 (rw-r--r--), Directories: 755 (rwxr-xr-x)
echo "→ Setting file permissions..."
chmod 644 ${OUTPUT_ZIP}

# Display package info
echo ""
echo "========================================================================"
echo "✓ Package built successfully!"
echo "========================================================================"
echo ""
ls -lh ${OUTPUT_ZIP}
echo ""

# Get file size in MB
size_bytes=$(stat -f%z ${OUTPUT_ZIP} 2>/dev/null || stat -c%s ${OUTPUT_ZIP} 2>/dev/null)
size_mb=$((size_bytes / 1024 / 1024))

echo "Package size: ${size_mb} MB"

# Check against limits
if [ ${size_mb} -gt 250 ]; then
    echo "⚠ WARNING: Package exceeds 250 MB limit (zipped)"
    echo "  Consider optimizing dependencies or removing unused packages"
fi

echo ""
echo "Next steps:"
echo "  1. Upload to S3:     aws s3 cp ${OUTPUT_ZIP} s3://your-bucket/path/"
echo "  2. Deploy agent:     uv run deploy.py --env staging"
echo ""
