#!/bin/bash

### Training partABC (001) Latent Diffusion Model ### 
# 경로: /mnt/sdb/TMEMJ/partABC_data/parsed_001
# 이미지 경로: /mnt/sdb/TMEMJ/partABC_data/abc_images

# 1. Surface Position 훈련
python ldm.py --data /mnt/sdb/TMEMJ/partABC_data/parsed_001 \
    --list /mnt/sdb/TMEMJ/partABC_data/parsed_001/split.pkl --option surfpos --gpu 0 1 \
    --env partABC_ldm_surfpos --train_nepoch 1000 --test_nepoch 200 --save_nepoch 200 \
    --max_face 50 --max_edge 30

# 2. Surface Z 훈련 (VAE 가중치 포함)
python ldm.py --data /mnt/sdb/TMEMJ/partABC_data/parsed_001 \
    --list /mnt/sdb/TMEMJ/partABC_data/parsed_001/split.pkl --option surfz \
    --surfvae proj_log/abc_vae_surf.pt --gpu 0 1 \
    --env partABC_ldm_surfz --train_nepoch 1000 --batch_size 256 \
    --max_face 50 --max_edge 30

# 3. Edge Position 훈련
python ldm.py --data /mnt/sdb/TMEMJ/partABC_data/parsed_001 \
    --list /mnt/sdb/TMEMJ/partABC_data/parsed_001/split.pkl --option edgepos \
    --surfvae proj_log/abc_vae_surf.pt --gpu 0 1 \
    --env partABC_ldm_edgepos --train_nepoch 300 --batch_size 64 \
    --max_face 50 --max_edge 30

# 4. Edge Z 훈련 (VAE 가중치 포함)
python ldm.py --data /mnt/sdb/TMEMJ/partABC_data/parsed_001 \
    --list /mnt/sdb/TMEMJ/partABC_data/parsed_001/split.pkl --option edgez \
    --surfvae proj_log/abc_vae_surf.pt --edgevae proj_log/abc_vae_edge.pt --gpu 0 1 \
    --env partABC_ldm_edgez --train_nepoch 300 --batch_size 64 \
    --max_face 50 --max_edge 30