#!/bin/bash
cd ~/BEPImitationlearning/zzgifs/
MAX=52428800  # 50MB in bytes

for f in *.webm; do
    output="gifs/${f%.webm}.gif"
    echo "Converting: $f"

    # Pass 1: standard quality
    ffmpeg -y -i "$f" \
      -vf "fps=10,scale=1200:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" \
      -loop 0 "$output" 2>/dev/null

    size=$(stat -c%s "$output")
    if [ $size -gt $MAX ]; then
        echo "  Too large ($(numfmt --to=iec $size)), trying pass 2..."
        ffmpeg -y -i "$f" \
          -vf "fps=8,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer" \
          -loop 0 "$output" 2>/dev/null
        size=$(stat -c%s "$output")
    fi

    if [ $size -gt $MAX ]; then
        echo "  Still too large ($(numfmt --to=iec $size)), trying pass 3..."
        ffmpeg -y -i "$f" \
          -vf "fps=6,scale=600:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=32[p];[s1][p]paletteuse=dither=bayer" \
          -loop 0 "$output" 2>/dev/null
        size=$(stat -c%s "$output")
    fi

    echo "  Done: $output — $(numfmt --to=iec $size)"
done

echo ""
echo "All done. Final sizes:"
ls -lh gifs/*.gif
