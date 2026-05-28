#!/bin/bash
# Ainamaura Checkers — push to GitHub
# Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME

if [ -z "$1" ]; then
    echo "Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME"
    exit 1
fi

USERNAME=$1
REPO="ainamaura-checkers"

cd "$(dirname "$0")"

git init
git add .
git commit -m "Ainamaura Checkers v1.0 — initial release"
git branch -M main
git remote add origin "https://github.com/$USERNAME/$REPO.git"
git push -u origin main

echo ""
echo "Done. Visit: https://github.com/$USERNAME/$REPO"
