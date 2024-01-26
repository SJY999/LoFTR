#!/bin/bash
set -e
# set -x

if [ ! -f utils.py ]; then
    echo "Downloading utils.py from the SuperGlue repo."
    echo "We cannot provide this file directly due to its strict licence."
    wget https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/utils.py
fi

# Use webcam 0 as input source. 
# input=0
# input=/home/sjy/TUM-RGBD/rgbd_dataset_freiburg1_desk2/rgb
# input=/home/sjy/Dataset/EuRoC/MH_01_easy/mav0/cam0/data
# input=/home/sjy/Dataset/EuRoC/V1_03_difficult/mav0/cam0/data
# input=/home/sjy/LoFTR/data/scannet/test/scene0707_00/color
# input=/home/sjy/LoFTR/data/scannet/test/scene0801_00/color
input=/home/sjy/LoFTR/data/scannet/test/scene0803_00/color  #失败了
# input=/home/sjy/LoFTR/data/scannet/test/scene0805_00/color
# or use a pre-recorded video given the path.
# input=/home/sunjiaming/Downloads/scannet_test/$scene_name.mp4

# Toggle indoor/outdoor model here.
model_ckpt=../weights/indoor_ds.ckpt

# model_ckpt=../weights/outdoor_ds.ckpt

# Optionally assign the GPU ID.
# export CUDA_VISIBLE_DEVICES=0
ways='loftr'

echo "Running LoFTR demo.."
eval "$(conda shell.bash hook)"
conda activate loftr

python demo_loftr.py --weight $model_ckpt --input $input --save_video --frame_interval=1 --way $ways

# python demo_loftr.py --weight $model_ckpt --input $input --save_video --frame_interval=20 --way $ways
# python demo_loftr.py --weight $model_ckpt --input $input --save_video --frame_interval=15 --way $ways
# python demo_loftr.py --weight $model_ckpt --input $input --save_video --frame_interval=25 --way $ways
# python demo_loftr.py --weight $model_ckpt --input $input --save_video --frame_interval=30 --way $ways
# python demo_loftr.py --weight $model_ckpt --input $input --save_video --frame_interval=40 --way $ways

# To save the input video and output match visualizations.
# python demo_loftr.py --weight $model_ckpt --input $input --save_video --save_input

# Running on remote GPU servers with no GUI.
# Save images first.
# python demo_loftr.py --weight $model_ckpt --input $input --no_display --output_dir="./demo_images/"
# Then convert them to a video.
# ffmpeg -framerate 15 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
