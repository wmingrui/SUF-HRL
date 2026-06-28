#!/usr/bin/env bash
set -e
# Run this inside the repository root after setting your GitHub remote.
# Example:
#   git init
#   git branch -M main
#   git remote add origin https://github.com/wmingrui/SUF-HRL.git
#   bash scripts/push_to_github.sh

git add .
git commit -m "Initial public release of SUF-HRL"
git push -u origin main
