#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=11
#SBATCH --job-name=mbart_shard
#SBATCH --nodelist=ilps-cn116
#SBATCH --time=20-00:00:00
#SBATCH --mem=250G
#SBATCH -o mbart50-ft.o
#SBATCH -e mbart50-ft.e

pwd
conda info --envs
source /home/stan1/anaconda3/bin/activate fairseq


# 1. download the mBart50 pretrained checkpoint
# reference: https://github.com/facebookresearch/fairseq/tree/main/examples/multilingual#mbart50-models
mbart_checkpoint=model_extended_EC40.pt

# 2. Extend the mBart50 vocabulary and embedding:
#    Because: a. If you add new unseen languages (beyond 50), then you need new tokens to specify those languages.
#             b. In addition, you need to expand the embedding such as encoder.embed_tokens.weight as well, e.g.: 
#                encoder.embed_tokens.weight.shape = (250054, 1024) -> (250054+n, 1024) n donotes the new added unseen languages.
#                to do that, you can random initialize new embedding weights or copy english weights -> both deliver similar results (in my case)
## run extend_mbart50.ipynb (adjust it according to your need )


# 3. Train
## remember to modify lang_dict first (add new unseen languages to ML50_langs_extended_EC40.txt)
fairseq-train mbart-fairseq-data-bin-sharded/shard0:mbart-fairseq-data-bin-sharded/shard1:mbart-fairseq-data-bin-sharded/shard2:mbart-fairseq-data-bin-sharded/shard3:mbart-fairseq-data-bin-sharded/shard4 \
  --task translation_multi_simple_epoch \
  --finetune-from-model $mbart_checkpoint \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --encoder-langtok src --decoder-langtok \
  --skip-invalid-size-inputs-valid-test \
  --sampling-method temperature --sampling-temperature 5 \
  --lang-dict ML50_langs_extended_EC40.txt \
  --lang-pairs en-de,en-nl,en-sv,en-da,en-is,en-af,en-lb,en-no,en-fr,en-es,en-it,en-pt,en-ro,en-oc,en-ast,en-ca,en-ru,en-cs,en-pl,en-bg,en-uk,en-sr,en-be,en-bs,en-hi,en-bn,en-kn,en-mr,en-sd,en-gu,en-ne,en-ur,en-ar,en-he,en-ha,en-mt,en-ti,en-am,en-kab,en-so,de-en,nl-en,sv-en,da-en,is-en,af-en,lb-en,no-en,fr-en,es-en,it-en,pt-en,ro-en,oc-en,ast-en,ca-en,ru-en,cs-en,pl-en,bg-en,uk-en,sr-en,be-en,bs-en,hi-en,bn-en,kn-en,mr-en,sd-en,gu-en,ne-en,ur-en,ar-en,he-en,ha-en,mt-en,ti-en,am-en,kab-en,so-en \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --max-update 100000 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 4096 --update-freq 7 \
  --save-interval-updates 2000 --keep-interval-updates 5 --no-epoch-checkpoints --log-interval 100 \
  --seed 1234 --patience 10 \
  --checkpoint-suffix _mbart_ --save-dir checkpoints/mbart50_shard \
  --wandb-project 'EC40' \
  --fp16 --ddp-backend no_c10d --distributed-world-size 6 --distributed-num-procs 66 --ddp-comm-hook fp16
