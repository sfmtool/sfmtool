#!/bin/bash
# Initialize Seoul Bull dataset workspace
# This script creates a workspace, copies test images, and sets up an sfm_solve.sh script
#
# Pipeline: sfmtool SIFT -> track-cluster matching (d=28) -> incremental SfM.
# The small 270x480 images use a wider cluster floor (d=28) so all 17 register.

set -e

WORKSPACE_DIR="seoul_bull_ws"
DATA_SOURCE="test-data/images/seoul_bull_sculpture"

echo "Initializing Seoul Bull dataset workspace..."

# Create workspace and images subdirectory
mkdir -p "${WORKSPACE_DIR}/images"

# Copy images from test data
echo "Copying Seoul Bull sculpture images..."
cp "${DATA_SOURCE}"/seoul_bull_sculpture_*.jpg "${WORKSPACE_DIR}/images/"

# Count images
NUM_IMAGES=$(ls -1 "${WORKSPACE_DIR}/images"/*.jpg | wc -l)
echo "Copied ${NUM_IMAGES} images to ${WORKSPACE_DIR}/images/"

# Create sfm_solve.sh script
cat > "${WORKSPACE_DIR}/sfm_solve.sh" << 'EOF'
#!/bin/bash
# SfM solve script for Seoul Bull dataset
# sfmtool SIFT -> track-cluster matching (d=28) -> incremental SfM

set -euo pipefail
cd "$(dirname "$0")"

# Initialize workspace with the sfmtool SIFT backend
if [ ! -f .sfm-workspace.json ]; then
  sfm ws init --feature-tool sfmtool .
fi

# Extract features with 3 threads
sfm sift --extract -t 3 images/*.jpg

# Track-cluster matching with a wider background floor (d=28)
mkdir -p tvg-matches
sfm match --cluster --cluster-d 28 images/ -o tvg-matches/seoul_bull.matches

echo "Running incremental SfM on Seoul Bull dataset..."
sfm solve --incremental --seed 42 tvg-matches/seoul_bull.matches

echo "Reconstruction complete! Check sfmr/ for results."
EOF

chmod +x "${WORKSPACE_DIR}/sfm_solve.sh"

echo ""
echo "Workspace initialized at: ${WORKSPACE_DIR}"
echo "To run reconstruction:"
echo "  ${WORKSPACE_DIR}/sfm_solve.sh"
echo "  # or, if the sfm command is not on your PATH:"
echo "  pixi run bash ${WORKSPACE_DIR}/sfm_solve.sh"
