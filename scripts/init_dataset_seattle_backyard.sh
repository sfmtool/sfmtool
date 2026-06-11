#!/bin/bash
# Initialize Seattle Backyard dataset workspace
# This script creates a workspace, copies test images, and sets up an sfm_solve.sh script
#
# Pipeline: sfmtool SIFT (capped at 2000 features) -> track-cluster matching ->
# global SfM (GLOMAP).

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
# sfmtool SIFT -> track-cluster matching -> global SfM (GLOMAP)

set -euo pipefail
cd "$(dirname "$0")"

# Initialize workspace with the sfmtool SIFT backend, capped at 2000 features
if [ ! -f .sfm-workspace.json ]; then
  sfm ws init --feature-tool sfmtool --max-features 2000 .
fi

# Extract features with 3 threads
sfm sift --extract -t 3 images/*.jpg

# Track-cluster matching
mkdir -p tvg-matches
sfm match --cluster images/ -o tvg-matches/seattle_backyard.matches

echo "Running global SfM (GLOMAP) on Seattle Backyard dataset..."
sfm solve --global --seed 42 tvg-matches/seattle_backyard.matches

echo "Reconstruction complete! Check sfmr/ for results."
EOF

chmod +x "${WORKSPACE_DIR}/sfm_solve.sh"

echo ""
echo "Workspace initialized at: ${WORKSPACE_DIR}"
echo "To run reconstruction:"
echo "  ${WORKSPACE_DIR}/sfm_solve.sh"
echo "  # or, if the sfm command is not on your PATH:"
echo "  pixi run bash ${WORKSPACE_DIR}/sfm_solve.sh"
