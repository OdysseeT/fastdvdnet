#!/bin/sh



rm -rf frames_in frames_out
mkdir frames_in frames_out
ffmpeg -i $1 -vf fps=24 frames_in/frame%d.png

python test_fastdvdnet.py --test_path frames_in --save_path frames_out --noise_sigma $2 --max_num_fr_per_seq $3


ffmpeg -framerate 24 \
    -pattern_type glob \
    -i 'frames_out/*.png' \
    -c:v libx264  \
    -r 30 \
    -pix_fmt yuv420p10le \
    -map_chapters 0 \
    -preset fast \
    -crf 21 \
    -c:a copy \
    $1-DENOISED-noise$2-seq-$3.mkv
