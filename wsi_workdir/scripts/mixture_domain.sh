# Modifiy the root to your own path
# e.g. root=/private/project/few-shot-wsi
root=/PATH/TO/YOUR/PROJECT/ROOT
cd $root


for model in clp fsp
do
    python3 wsi_workdir/dict_construction.py \
        --model ${model} \
        
    for clf in Ridge logistic_regression
    do
        date
        for num_shot in 1 5 10
        do
            python3 -u wsi_workdir/distributed_meta_test.py \
                --task mixture   \
                --num_task 1000  \
                --num_shots ${num_shot}   \
                --mode linear   \
                --model $model \
                --clf ${clf}

            python3 -u wsi_workdir/distributed_meta_test.py \
                --task mixture   \
                --num_task 1000  \
                --num_shots ${num_shot}   \
                --mode latent_aug   \
                --model $model \
                --clf ${clf}
        done
    done



    for num_shot in 1 5 10
    do
        python3 -u wsi_workdir/distributed_meta_test.py \
            --task mixture   \
            --num_task 1000  \
            --num_shots ${num_shot}   \
            --mode linear   \
            --model $model \
            --clf nearest_centroid
    done
done