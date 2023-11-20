#!/bin/bash

cd ZS-NMT-Variations/get-val-test-data

TOOLDIR=tools
SPMDIR=spm_dict # the spm model and vocab, and fairseq dict directory (place it if you use your own vocab)
num_cpus=20 #change this to your num_cpus
clean_script_dir=$TOOLDIR/data_preprocess/clean_scripts
tokenize_script_dir=$TOOLDIR/data_preprocess/tokenize_scripts


### set up if you encountered perl: warning: Please check that your locale settings: LANGUAGE = (unset), LC_ALL = (unset) ...
export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8

# ------------------------ 1. Extract NTREX raw data ------------------------ #

NTREX_DIR=NTREX/NTREX-128
EC40_VAL_DIR=EC40-NTREX-val/raw/eval_en_data
EC40_ADD_VAL_DIR=val-others

git clone https://github.com/MicrosoftTranslator/NTREX.git

# Ntrex-128 urd file contains some empty lines (17 Nov 2023)
python $TOOLDIR/remove_empty_line_ntrex.py --ntrex-dir $NTREX_DIR

## extract the NTREX data we need for EC40
python $TOOLDIR/extract_ntrex_ec40.py --ntrex-dir $NTREX_DIR --output-dir $EC40_VAL_DIR

# we lack three languages (oc,ast,kab) in Ntrex, EC40 provide the sampled val data for them 
cp -a $EC40_ADD_VAL_DIR/. $EC40_VAL_DIR/.
echo "Copy adiidtional val data Done!"

# ------------------------ 2. Normalize & Apply BPE En-centric ------------------------ #
RAW_DATA_DIR=$EC40_VAL_DIR 
TOK_DATA_DIR=EC40-NTREX-val/tok/eval_en_data
SPM_DATA_DIR=EC40-NTREX-val/spm/eval_en_data

mkdir -p $TOK_DATA_DIR
mkdir -p $SPM_DATA_DIR

SPM_MODEL_DIR=$SPMDIR/spm_64k.model
SPM_VOCAB_DIR=$SPMDIR/spm_64k.txt

AR=('de' 'nl' 'sv' 'da' 'is' 'af' 'lb' 'no' 'fr' 'es' 'it' 'pt' 'ro' 'oc' 'ast' 'ca' 'ru' 'cs' 'pl' 'bg' 'uk' 'sr' 'be' 'bs' 'hi' 'bn' 'kn' 'mr' 'sd' 'gu' 'ne' 'ur' 'ar' 'he' 'ha' 'mt' 'ti' 'am' 'kab' 'so')
for i in "${!AR[@]}"; do
    TGT=${AR[i]}
    SRC=en

    ## ------------------------ 2.1. clean ------------------------ #
    #  ------- normalize-punctuation -------
    cat $RAW_DATA_DIR/test.$SRC-$TGT.$SRC | perl ${clean_script_dir}/normalize-punctuation.pl -l ${SRC} -threads ${num_cpus} > $TOK_DATA_DIR/test.$SRC-$TGT.$SRC.norm
    cat $RAW_DATA_DIR/test.$SRC-$TGT.$TGT | perl ${clean_script_dir}/normalize-punctuation.pl -l ${TGT} -threads ${num_cpus} > $TOK_DATA_DIR/test.$SRC-$TGT.$TGT.norm

    #### ------------------------ 2.2. Tokenize ------------------------ #
    cat $TOK_DATA_DIR/test.$SRC-$TGT.$SRC.norm | perl ${tokenize_script_dir}/moses_tokenizer.pl -a -q -l ${SRC} -no-escape -threads ${num_cpus} > $TOK_DATA_DIR/test.$SRC-$TGT.$SRC
    cat $TOK_DATA_DIR/test.$SRC-$TGT.$TGT.norm | perl ${tokenize_script_dir}/moses_tokenizer.pl -a -q -l ${TGT} -no-escape -threads ${num_cpus} > $TOK_DATA_DIR/test.$SRC-$TGT.$TGT

done
find ${TOK_DATA_DIR} -type f -name "*.norm" | xargs rm

#### ------------------------ 2.3. Apply SPM ------------------------ #
for curlang in bg so ca da be bs mt es uk am hi ro no ti de cs lb pt nl mr is ne ur oc ast ha sv kab gu ar fr ru it pl sr sd he af kn bn; do
    spm_encode --model ${SPM_MODEL_DIR} < ${TOK_DATA_DIR}/test.en-${curlang}.en > ${SPM_DATA_DIR}/test.en-${curlang}.en
    spm_encode --model ${SPM_MODEL_DIR} < ${TOK_DATA_DIR}/test.en-${curlang}.${curlang} > ${SPM_DATA_DIR}/test.en-${curlang}.${curlang}
    
done
# ------------------------ 3. Fairseq preparation ------------------------ #
FAIRSEQ_DATA_DIR=EC40-NTREX-val/fairseq/eval_en_data
FAIRSEQ_VOCAB_DIR=$SPMDIR/fairseq_dict.txt

mkdir -p $FAIRSEQ_DATA_DIR

for SRC in en; do
    for TGT in bg so ca da be bs mt es uk am hi ro no ti de cs lb pt nl mr is ne ur oc ast ha sv kab gu ar fr ru it pl sr sd he af kn bn; do
        fairseq-preprocess \
        --source-lang ${SRC} \
        --target-lang ${TGT} \
        --srcdict ${FAIRSEQ_VOCAB_DIR} \
        --tgtdict ${FAIRSEQ_VOCAB_DIR} \
        --validpref ${SPM_DATA_DIR}/test.$SRC-$TGT \
        --destdir ${FAIRSEQ_DATA_DIR} \
        --workers ${num_cpus}
    done
done

mv EC40-NTREX-val/fairseq/eval_en_data EC40-NTREX-val-data-bin
rm -R EC40-NTREX-val #keep if you need non-fairseq data
rm -R NTREX

echo "Done ALL!"