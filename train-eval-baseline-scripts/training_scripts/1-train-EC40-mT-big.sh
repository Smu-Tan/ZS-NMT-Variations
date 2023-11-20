#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=11
#SBATCH --job-name=m2m_base_shard
#SBATCH --nodelist=ilps-cn116
#SBATCH --time=20-00:00:00
#SBATCH --mem=250G
#SBATCH -o mt-large.o
#SBATCH -e mt-large.e

pwd
conda info --envs
source /home/stan1/anaconda3/bin/activate fairseq


fairseq-train fairseq-data-bin-sharded/shard0:fairseq-data-bin-sharded/shard1:fairseq-data-bin-sharded/shard2:fairseq-data-bin-sharded/shard3:fairseq-data-bin-sharded/shard4 \
    --langs en,de,nl,sv,da,is,af,lb,no,fr,es,it,pt,ro,oc,ast,ca,ru,cs,pl,bg,uk,sr,be,bs,hi,bn,kn,mr,sd,gu,ne,ur,ar,he,ha,mt,ti,am,kab,so \
    --lang-pairs en-de,en-nl,en-sv,en-da,en-is,en-af,en-lb,en-no,en-fr,en-es,en-it,en-pt,en-ro,en-oc,en-ast,en-ca,en-ru,en-cs,en-pl,en-bg,en-uk,en-sr,en-be,en-bs,en-hi,en-bn,en-kn,en-mr,en-sd,en-gu,en-ne,en-ur,en-ar,en-he,en-ha,en-mt,en-ti,en-am,en-kab,en-so,de-en,nl-en,sv-en,da-en,is-en,af-en,lb-en,no-en,fr-en,es-en,it-en,pt-en,ro-en,oc-en,ast-en,ca-en,ru-en,cs-en,pl-en,bg-en,uk-en,sr-en,be-en,bs-en,hi-en,bn-en,kn-en,mr-en,sd-en,gu-en,ne-en,ur-en,ar-en,he-en,ha-en,mt-en,ti-en,am-en,kab-en,so-en \
    --task translation_multi_simple_epoch \
    --encoder-langtok tgt \
    --arch transformer_vaswani_wmt_en_de_big \
    --encoder-normalize-before --decoder-normalize-before --layernorm-embedding \
    --encoder-layers 6 --decoder-layers 6 \
    --sampling-method temperature --sampling-temperature 5 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 10240 --update-freq 21 --max-update 900000 \
    --share-all-embeddings \
    --max-source-positions 256 --max-target-positions 256 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --seed 1234 --patience 10 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --weight-decay 0.0 \
    --dropout 0.1 --attention-dropout 0.1 \
    --fp16 --ddp-backend no_c10d \
    --checkpoint-suffix _m2m_ --save-dir checkpoints/mt_big \
    --save-interval-updates 2000 --keep-interval-updates 5 --no-epoch-checkpoints --log-interval 100 \
    --distributed-world-size 4 --distributed-num-procs 44 --ddp-comm-hook fp16