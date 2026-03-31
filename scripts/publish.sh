#!/bin/bash
FILE=$(echo "$1" | sed 's/\\//g')
BLOG=$(cd "$(dirname "$0")/.." && pwd)
FILENAME=$(basename "$FILE")

cp "$FILE" "$BLOG/content/posts/$FILENAME"
cd "$BLOG"
git add -A
git commit -m "publish: $FILENAME"
git push
