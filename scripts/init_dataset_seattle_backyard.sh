#!/bin/bash
# Initialize Seattle Backyard dataset workspace
# This script creates a workspace, copies test images, and sets up an sfm_solve.sh script

set -e

WORKSPACE_DIR="seattle_backyard_ws"
DATA_SOURCE="test-data/images/seattle_backyard"

echo "Initializing Seattle Backyard dataset workspace..."

# Create workspace and images subdirectory
mkdir -p "${WORKSPACE_DIR}/images"

# Copy images from test data
echo "Copying Seattle Backyard images..."
cp "${DATA_SOURCE}"/seattle_backyard_*.jpg "${WORKSPACE_DIR}/images/"

# Count images
NUM_IMAGES=$(ls -1 "${WORKSPACE_DIR}/images"/*.jpg | wc -l)
echo "Copied ${NUM_IMAGES} images to ${WORKSPACE_DIR}/images/"

# Create sfm_solve.sh script
cat > "${WORKSPACE_DIR}/sfm_solve.sh" << 'EOF'
#!/bin/bash
# SfM solve script for Seattle Backyard dataset
# Uses global SfM with GLOMAP, 250 max features

set -euo pipefail
cd "$(dirname "$0")"

# Initialize workspace
if [ ! -f .sfm-workspace.json ]; then
  sfm ws init .
fi

# Extract features with 3 threads
sfm sift --extract -t 3 images/*.jpg

echo "Running global SfM (GLOMAP) on Seattle Backyard dataset..."
sfm solve --global --max-features 250 --seed 42 images/

echo "Reconstruction complete! Check sfmr/ for results."
EOF

chmod +x "${WORKSPACE_DIR}/sfm_solve.sh"

echo ""
echo "Workspace initialized at: ${WORKSPACE_DIR}"
echo "To run reconstruction:"
echo "  ${WORKSPACE_DIR}/sfm_solve.sh"
echo "  # or, if the sfm command is not on your PATH:"
echo "  pixi run bash ${WORKSPACE_DIR}/sfm_solve.sh"
