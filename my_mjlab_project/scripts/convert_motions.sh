#!/usr/bin/env bash
# Convert all 6 goalkeeper motion clips from .pt to mjlab NPZ format.
# Run from /home/isaak/BEPImitationlearning/my_mjlab_project/
set -e

INPUT=/home/isaak/BEPImitationlearning/Humanoid-Goalkeeper/legged_gym/resources/datasets/goalkeeper
OUTPUT=src/my_mjlab_project/motions/data

echo "Converting motion files from $INPUT → $OUTPUT"
uv run python -m my_mjlab_project.motions.convert "$INPUT" "$OUTPUT"

echo ""
echo "Done. Files in $OUTPUT:"
ls -lh "$OUTPUT"
