#!/bin/bash
set -e

echo "=== Starting Repository Sync ==="

cd /workspaces/SDP_Agri_Swastha

# Fetch latest changes
echo "Fetching latest changes..."
git fetch origin

# Try to pull changes
echo "Pulling changes from remote..."
git pull origin main --allow-unrelated-histories || {
    echo "Pull failed, trying alternative approach..."
    git fetch origin
    git reset --hard origin/main
}

echo "=== Repository sync completed ==="
git status

