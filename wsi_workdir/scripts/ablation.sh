
# Modifiy the root to your own path
# e.g. root=/private/project/few-shot-wsi
root=/PATH/TO/YOUR/PROJECT/ROOT
cd $root

date
echo "##### Ablation on number of prototypes in base dictionary  #########"
echo "The novel classes are 7 STR and 8 TUM"

date
echo "Baseline:"
python3 -u wsi_workdir/distributed_meta_test.py \
    --task NCT_78_aug   \
    --num_task 300  \
    --num_shots 5   \
    --mode linear  \
    --model clp_wo_78 \

echo "Latent augmentation (aug 100 times):"
for num_proto in 2 4 8 16 32 64 128 256
do
    python3 -u wsi_workdir/dict_construction.py \
        --model clp_wo_78 \
        --num_proto $num_proto \
        --novel_class 78
    
    python3 -u wsi_workdir/distributed_meta_test.py \
        --task NCT_78_aug   \
        --num_task 300  \
        --num_shots 5   \
        --mode latent_aug  \
        --model clp_wo_78 \
        --num_prototypes $num_proto
done

echo "Done"




date
echo "##### Ablation on Date Augmentation v.s. Latent Augmentation, and Augmentation times  #########"
echo "The novel classes are 7 STR and 8 TUM"
echo "#### Data Augmentation:"
for aug_times in {2..16}
do
    python3 -u wsi_workdir/distributed_meta_test_wt_aug.py \
        --mode linear \
        --aug_times $((aug_times * aug_times)) \
        --num_task 300 \
        --model clp_wo_78 \
        --task NCT_78_aug
done

date
echo "Latent Augmentation:"
for aug_times in {2..16}
do
    python3 -u wsi_workdir/distributed_meta_test.py \
        --mode latent_aug \
        --num_aug_shots $((aug_times * aug_times)) \
        --model clp_wo_78 \
        --task NCT_78_aug \
        --num_task 300
done


date
echo "Data Augmentation \times Latent Augmentation:"
for aug_times in {2..16}
do 
    python3 -u wsi_workdir/distributed_meta_test_wt_aug.py \
        --mode latent_aug \
        --aug_times $aug_times \
        --num_aug_shots $aug_times \
        --model clp_wo_78
done


