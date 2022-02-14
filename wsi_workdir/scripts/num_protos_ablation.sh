# Modifiy the root to your own path
# e.g. root=/private/project/few-shot-wsi
root=/PATH/TO/YOUR/PROJECT/ROOT
cd $root

for num_proto in 2 4 8 16 32 64 128 256
do
    python3 -u submission_wsi/dict_construction.py \
        --model clp_wo_78 \
        --num_proto $num_proto \
        --novel_class 78
    
    python3 -u submission_wsi/distributed_meta_test.py \
        --task NCT_78_aug   \
        --num_task 300  \
        --num_shots 5   \
        --mode latent_aug  \
        --model clp_wo_78 \
        --num_prototypes $num_proto
done

