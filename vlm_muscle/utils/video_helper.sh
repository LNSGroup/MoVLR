MODEL_PATH="$1"
DATA_PATH="qpos $2"
MODE="kinematics"
OUTPUT="$3"
RESOLUTION="1080p"
CAMERA=$4
RECORD_DATA=0
RECORD_VIDEO=1
DATA_FORMAT="npy"
RECORD_TYPES="qpos qvel xpos"
# ACTIVATION=""
# ACTIVATION SHAPE 10 9

# Optional variables
if [ -n "$5" ]; then
    FPS="$5"
else
    FPS=500
fi

if [ -n "$6" ]; then
    PREFIX="$6"
else
    PREFIX="output"
fi

# Build command
CMD="mujoco-tools \\
    -m \"$MODEL_PATH\" \\
    -d \"$DATA_PATH\" \\
    --mode \"$MODE\" \\
    --output_prefix \"$PREFIX\" \\
    --output_path \"$OUTPUT\" \\
    --fps \"$FPS\" \\
    --camera \"$CAMERA\""

# Add recording options
if [ "$RECORD_DATA" -eq 1 ] && [ "$RECORD_VIDEO" -eq 1 ]; then
    CMD+=" --record_data"
    CMD+=" --record_video"
    CMD+=" --format \"$DATA_FORMAT\""
    CMD+=" --datatype \"$RECORD_TYPES\""
elif [ "$RECORD_DATA" -eq 1 ]; then
    CMD+=" --record_data"
    CMD+=" --format \"$DATA_FORMAT\""
    CMD+=" --datatype \"$RECORD_TYPES\""
elif [ "$RECORD_VIDEO" -eq 1 ]; then
    CMD+=" --record_video"
fi

# Execute command
eval "$CMD"

# Convert video to 100 fps for vlm, but keep the original video speed etc
if [ -s "$OUTPUT/${PREFIX}_video.mp4" ]; then
    ffmpeg -i "$OUTPUT/${PREFIX}_video.mp4" -filter:v "fps=25" "$OUTPUT/${PREFIX}_video_final.mp4"
    echo "Video processing complete. Output saved to $OUTPUT/${PREFIX}_video_final.mp4"
else
    echo "Error: Input video file $OUTPUT/${PREFIX}_video.mp4 does not exist or is empty."
fi