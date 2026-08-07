from instant_nsr_pl.launch import main as neus_main

neus_main({
    "gpu": "0",
    "config": "neus/configs/neuralangelo-ortho-wmask.yaml",
    "dataset": [
        "dataset.root_dir=/mnt/sdb/TMEMJ/CADDreamer-jjy/test_outputs",
        "dataset.scene=/mnt/sdb/TMEMJ/CADDreamer-jjy/test_outputs/cropsize-256-cfg3.0/2_0deepcad",
    ],
})
