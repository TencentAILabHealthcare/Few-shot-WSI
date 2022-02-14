
# out=out-domain, heterogeneous shot selection
# out_homo = out-domain, homogeneous shot selection. 
for task in out out_homo
do
    # uncomment below if you haven't generated dictionaries
    # python3 wsi_workdir/dict_construction.py \
    #     --model ${model} \

    date
    for num_shot in 1 5 10 50 100
    do
        python3 -u wsi_workdir/distributed_meta_test.py \
            --task ${task}   \
            --num_task 1000  \
            --num_shots ${num_shot}   \
            --mode linear   \
            --model fsp \
            --clf Ridge
            
        python3 -u wsi_workdir/distributed_meta_test.py \
            --task ${task}   \
            --num_task 1000  \
            --num_shots ${num_shot}   \
            --mode linear   \
            --model clp \
            --clf Ridge

        python3 -u wsi_workdir/distributed_meta_test.py \
            --task ${task}   \
            --num_task 1000  \
            --num_shots ${num_shot}   \
            --mode latent_aug   \
            --model clp \
            --clf Ridge
        
        
    done
done