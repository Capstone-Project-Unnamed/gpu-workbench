conda activate /mnt/sdb/envs/caddreamer-jjy
cd /mnt/sdb/TMEMJ/CADDreamer-jjy
python test_mvdiffusion_seq.py \
    --config configs/mvdiffusion-joint-ortho-6views.yaml \
    --idx 0 \
    --gpu 0

python3 test_real_images.py \
    --config ./cached_output/cropsize-256-cfg1.0/0_0deepcad \
    --review False

# step 1
LD_LIBRARY_PATH=/mnt/sdb/envs/caddreamer-jjy/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/mnt/sdb/TMEMJ/CADDreamer-jjy/pyransac/cmake-build-release:/mnt/sdb/TMEMJ/CADDreamer-jjy \
python3 test_mvdiffusion_seq.py \
    --config configs/train/testing_4090_stage_1_cad_6views-lvis.yaml \
    --idx 0 --gpu 0
# step 2
LD_LIBRARY_PATH=/mnt/sdb/envs/caddreamer-jjy/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/mnt/sdb/TMEMJ/CADDreamer-jjy/pyransac/cmake-build-release:/mnt/sdb/TMEMJ/CADDreamer-jjy \
python3 test_real_images.py --config ./test_outputs/cropsize-256-cfg1.0/0_0deepcad --review False