rm snapshots/*

export NAME=$(basename "$PWD")

nvidia-docker rm -f $NAME

NV_GPU=1 nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups:/groups \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.2 \
    python -u train.py
