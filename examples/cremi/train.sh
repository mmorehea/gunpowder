rm snapshots/*

export NAME=$(basename "$PWD")

nvidia-docker rm $NAME

NV_GPU=0 nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:latest \
    python -u cremi.py