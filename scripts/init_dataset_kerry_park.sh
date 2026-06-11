#!/bin/bash
# Initialize Kerry Park dataset workspace
# This script creates a workspace, copies the rig images and rig_config.json,
# and sets up an sfm_solve.sh script
#
# Pipeline: sfmtool SIFT -> track-cluster matching -> global SfM (GLOMAP) on a
# 360-degree fisheye rig.

set -e

WORKSPACE_DIR="kerry_park_ws"
DATA_SOURCE="test-data/images/kerry_park"

echo "Initializing Kerry Park dataset workspace..."

# Create workspace and rig subdirectories
mkdir -p "${WORKSPACE_DIR}/fisheye_left"
mkdir -p "${WORKSPACE_DIR}/fisheye_right"

# Copy images and rig configuration
echo "Copying Kerry Park fisheye images..."
cp "${DATA_SOURCE}"/fisheye_left/frame_*.jpg "${WORKSPACE_DIR}/fisheye_left/"
cp "${DATA_SOURCE}"/fisheye_right/frame_*.jpg "${WORKSPACE_DIR}/fisheye_right/"
cp "${DATA_SOURCE}/rig_config.json" "${WORKSPACE_DIR}/rig_config.json"

# Count frames (each fisheye_{left,right}/frame_NN.jpg is one rig frame)
NUM_FRAMES=$(ls -1 "${WORKSPACE_DIR}/fisheye_left"/*.jpg | wc -l)
echo "Copied ${NUM_FRAMES} rig frames to ${WORKSPACE_DIR}/"

# Create sfm_solve.sh script
cat > "${WORKSPACE_DIR}/sfm_solve.sh" << 'EOF'
#!/bin/bash
# SfM solve script for Kerry Park dataset
# sfmtool SIFT -> track-cluster matching -> global SfM (GLOMAP) on a 360-degree
# fisheye rig

set -euo pipefail
cd "$(dirname "$0")"

# Initialize workspace with the sfmtool SIFT backend
if [ ! -f .sfm-workspace.json ]; then
  sfm ws init --feature-tool sfmtool .
fi

# Extract features for both fisheye sensors
sfm sift --extract -t 3 fisheye_left/*.jpg fisheye_right/*.jpg

# Track-cluster matching
mkdir -p tvg-matches
sfm match --cluster fisheye_left fisheye_right -o tvg-matches/kerry_park.matches

echo "Running global SfM (GLOMAP) on Kerry Park dataset..."
sfm solve --global --seed 42 tvg-matches/kerry_park.matches

echo "Reconstruction complete! Check sfmr/ for results."
EOF

chmod +x "${WORKSPACE_DIR}/sfm_solve.sh"

echo ""
echo "Workspace initialized at: ${WORKSPACE_DIR}"
echo "To run reconstruction:"
echo "  ${WORKSPACE_DIR}/sfm_solve.sh"
echo "  # or, if the sfm command is not on your PATH:"
echo "  pixi run bash ${WORKSPACE_DIR}/sfm_solve.sh"
