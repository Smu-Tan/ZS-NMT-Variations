#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --job-name=process
#SBATCH --nodelist=ilps-cn002
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G

pwd
conda info --envs
source /home/stan1/anaconda3/bin/activate fairseq
cd /ivi/ilps/projects/ltl-mt/EC40-dataset

mkdir spm_sharded


######################## ------------ IMPORTRANT ------------ ########################

######## This is an example of how to build a sharded dataset (5 shards)
######## Before run the following code, you should have trained your sentencepiece/subword-mt tokenizer already
######## Then you should encode the dataset using spm, and then use following code to split them to 5 shards

#### For eval set, the most easiest way is to add the whole eval-set to all 5 shard fairseq data folder
### note: ha and kab is two exceptions (because of their data-size): you will find them in *SPECIAL* 

######################## ------------ IMPORTRANT ------------ ########################

SHARD_SUB_DIR=('0' '1' '2' '3' '4')
for i in "${!SHARD_SUB_DIR[@]}"; do
    SUB_NUMBER=${SHARD_SUB_DIR[i]}
    mkdir dataset/spm_sharded/shard${SUB_NUMBER}
done

HIGH=('de' 'nl' 'fr' 'es' 'ru' 'cs' 'hi' 'bn' 'ar' 'he')
MED=('sv' 'da' 'it' 'pt' 'pl' 'bg' 'kn' 'mr' 'mt') #ha
LOW=('af' 'lb' 'ro' 'oc' 'uk' 'sr' 'sd' 'gu' 'ti' 'am')
ELOW=('no' 'is' 'ast' 'ca' 'be' 'bs' 'ne' 'ur' 'so') #kab

SPM_DIR=dataset/spm
SPM_SHARD_DIR=dataset/spm_sharded

##

## HIGH 5m each file -> split to 1m for one shard
for i in "${!HIGH[@]}"; do
    LANG=${HIGH[i]}
    split -l 1000000 $SPM_DIR/train.en-$LANG.en -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.en.shard
    split -l 1000000 $SPM_DIR/train.en-$LANG.$LANG -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard

    for j in "${!SHARD_SUB_DIR[@]}"; do
        SUB_NUMBER=${SHARD_SUB_DIR[j]}
        mv $SPM_SHARD_DIR/train.en-$LANG.en.shard0${SUB_NUMBER} dataset/spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.en
        mv $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard0${SUB_NUMBER} dataset/spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.$LANG
    done
done

# MED 1m each file -> split to 200K for one shard
for i in "${!MED[@]}"; do
    LANG=${MED[i]}
    split -l 200000 $SPM_DIR/train.en-$LANG.en -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.en.shard
    split -l 200000 $SPM_DIR/train.en-$LANG.$LANG -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard

    for j in "${!SHARD_SUB_DIR[@]}"; do
        SUB_NUMBER=${SHARD_SUB_DIR[j]}
        mv $SPM_SHARD_DIR/train.en-$LANG.en.shard0${SUB_NUMBER} dataset/spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.en
        mv $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard0${SUB_NUMBER} dataset/spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.$LANG
    done
done

# LOW 100k each file -> split to 20k for one shard
for i in "${!LOW[@]}"; do
    LANG=${LOW[i]}
    split -l 20000 $SPM_DIR/train.en-$LANG.en -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.en.shard
    split -l 20000 $SPM_DIR/train.en-$LANG.$LANG -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard

    for j in "${!SHARD_SUB_DIR[@]}"; do
        SUB_NUMBER=${SHARD_SUB_DIR[j]}
        mv $SPM_SHARD_DIR/train.en-$LANG.en.shard0${SUB_NUMBER} dataset/spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.en
        mv $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard0${SUB_NUMBER} dataset/spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.$LANG
    done
done

## ELOW 50k each file -> split to 10k for one shard
for i in "${!ELOW[@]}"; do
    LANG=${ELOW[i]}
    split -l 10000 $SPM_DIR/train.en-$LANG.en -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.en.shard
    split -l 10000 $SPM_DIR/train.en-$LANG.$LANG -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard

    for j in "${!SHARD_SUB_DIR[@]}"; do
        SUB_NUMBER=${SHARD_SUB_DIR[j]}
        mv $SPM_SHARD_DIR/train.en-$LANG.en.shard0${SUB_NUMBER} dataset/spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.en
        mv $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard0${SUB_NUMBER} dataset/spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.$LANG
    done
done

# SPECIAL HA 344000 -> split to 68800 for one shard 
HA=('ha')
for i in "${!HA[@]}"; do
    LANG=${HA[i]}
    split -l 68800 $SPM_DIR/train.en-$LANG.en -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.en.shard
    split -l 68800 $SPM_DIR/train.en-$LANG.$LANG -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard

    for j in "${!SHARD_SUB_DIR[@]}"; do
        SUB_NUMBER=${SHARD_SUB_DIR[j]}
        mv $SPM_SHARD_DIR/train.en-$LANG.en.shard0${SUB_NUMBER} dataset/spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.en
        mv $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard0${SUB_NUMBER} dataset/spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.$LANG
    done
done

# SPECIAL HA 18448 -> split to 3690 for one shard 
KAB=('kab')
for i in "${!KAB[@]}"; do
    LANG=${KAB[i]}
    split -l 3690 $SPM_DIR/train.en-$LANG.en -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.en.shard
    split -l 3690 $SPM_DIR/train.en-$LANG.$LANG -d -a 2 $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard

    for j in "${!SHARD_SUB_DIR[@]}"; do
        SUB_NUMBER=${SHARD_SUB_DIR[j]}
        mv $SPM_SHARD_DIR/train.en-$LANG.en.shard0${SUB_NUMBER} dataset/spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.en
        mv $SPM_SHARD_DIR/train.en-$LANG.$LANG.shard0${SUB_NUMBER} dataset/spm_sharded/shard${SUB_NUMBER}/train.en-$LANG.$LANG
    done
done

# ------------------------ 4. Fairseq preparation Sharded ------------------------ #
SPM_DATA_DIR=dataset/spm_sharded
FAIRSEQ_DIR=dataset/fairseq-data-bin-sharded
mkdir ${FAIRSEQ_DIR}

cut -f1 dataset/spm/spm_64k.vocab | tail -n +4 | sed "s/$/ 100/g" > ${FAIRSEQ_DIR}/dict.txt

SHARD_SUB_DIR=('0' '1' '2' '3' '4')
for i in "${!SHARD_SUB_DIR[@]}"; do
    SUB_NUMBER=${SHARD_SUB_DIR[i]}
    mkdir $FAIRSEQ_DIR/shard${SUB_NUMBER}
done

# preprocess with mmap dataset
for SHARD in $(seq 0 4); do
    SRC=en
    for TGT in bg so ca da be bs mt es uk am hi ro no ti de cs lb pt nl mr is ne ur oc ast ha sv kab gu ar fr ru it pl sr sd he af kn bn; do
        fairseq-preprocess \
            --dataset-impl mmap \
            --source-lang ${SRC} \
            --target-lang ${TGT} \
            --trainpref ${SPM_DATA_DIR}/shard${SHARD}/train.${SRC}-${TGT} \
            --destdir ${FAIRSEQ_DIR}/shard${SHARD} \
            --thresholdtgt 0 \
            --thresholdsrc 0 \
            --workers 40 \
            --srcdict ${FAIRSEQ_DIR}/dict.txt \
            --tgtdict ${FAIRSEQ_DIR}/dict.txt
    cp ${FAIRSEQ_DIR}/dict.txt ${FAIRSEQ_DIR}/shard${SHARD}/dict.txt
    done
done