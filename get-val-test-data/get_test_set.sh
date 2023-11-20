#!/bin/bash

cd ZS-NMT-Variations/get-val-test-data

TOOLDIR=tools
SPMDIR=spm_dict # the spm model and vocab, and fairseq dict directory (place it if you use your own vocab)
num_cpus=40 #change this to your num_cpus
clean_script_dir=$TOOLDIR/data_preprocess/clean_scripts
tokenize_script_dir=$TOOLDIR/data_preprocess/tokenize_scripts

### set up if you encountered perl: warning: Please check that your locale settings: LANGUAGE = (unset), LC_ALL = (unset) ...
export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8


# ------------------------ 1. Extract FLORES200 raw data ------------------------ #

FLORES_DIR=flores200_dataset
EC40_TEST_DIR=EC40-FLORES-test/raw

wget --trust-server-names https://tinyurl.com/flores200dataset
tar -xvf flores200_dataset.tar.gz

rm flores200_dataset.tar.gz

# convert serbian cyrilic to latin
cyrtranslit -l sr -i flores200_dataset/dev/srp_Cyrl.dev -o flores200_dataset/dev/srp_Latn.dev
cyrtranslit -l sr -i flores200_dataset/devtest/srp_Cyrl.devtest -o flores200_dataset/devtest/srp_Latn.devtest

echo "convert serbian cyrilic to latin done"

# merge flores200 
mkdir flores200_dataset/merged
for LANG in eng_Latn deu_Latn nld_Latn swe_Latn dan_Latn isl_Latn afr_Latn ltz_Latn nob_Latn fra_Latn spa_Latn ita_Latn por_Latn ron_Latn oci_Latn ast_Latn cat_Latn rus_Cyrl ces_Latn pol_Latn bul_Cyrl ukr_Cyrl srp_Latn bel_Cyrl bos_Latn hin_Deva ben_Beng kan_Knda mar_Deva snd_Arab guj_Gujr npi_Deva urd_Arab arb_Arab heb_Hebr hau_Latn mlt_Latn tir_Ethi amh_Ethi som_Latn kab_Latn; do
    cat flores200_dataset/dev/${LANG}.dev flores200_dataset/devtest/${LANG}.devtest >> flores200_dataset/merged/${LANG}.flores200
done

echo "merge done"

python $TOOLDIR/extract_flores200_ec40.py --flores-dir $FLORES_DIR --output-dir $EC40_TEST_DIR

rm -R $FLORES_DIR

# ------------------------ 2. Normalize & Apply BPE & Fairseq En-centric ------------------------ #
RAW_DATA_DIR=EC40-FLORES-test/raw/test_en_data 
TOK_DATA_DIR=EC40-FLORES-test/tok/test_en_data 
SPM_DATA_DIR=EC40-FLORES-test/spm/test_en_data
FAIRSEQ_DATA_DIR=EC40-FLORES-test/fairseq/test_en_data

mkdir -p $TOK_DATA_DIR
mkdir -p $SPM_DATA_DIR
mkdir -p $FAIRSEQ_DATA_DIR

SPM_MODEL_DIR=$SPMDIR/spm_64k.model
SPM_VOCAB_DIR=$SPMDIR/fairseq_dict.txt

AR=('de' 'nl' 'sv' 'da' 'is' 'af' 'lb' 'no' 'fr' 'es' 'it' 'pt' 'ro' 'oc' 'ast' 'ca' 'ru' 'cs' 'pl' 'bg' 'uk' 'sr' 'be' 'bs' 'hi' 'bn' 'kn' 'mr' 'sd' 'gu' 'ne' 'ur' 'ar' 'he' 'ha' 'mt' 'ti' 'am' 'kab' 'so')
for i in "${!AR[@]}"; do
    TGT=${AR[i]}
    SRC=en

    ### ------------------------ 2.1. clean ------------------------ #

    ##  ------- normalize-punctuation -------
    cat $RAW_DATA_DIR/test.$SRC-$TGT.$SRC | perl ${clean_script_dir}/normalize-punctuation.pl -l ${SRC} -threads ${num_cpus} > $TOK_DATA_DIR/test.$SRC-$TGT.$SRC.norm
    cat $RAW_DATA_DIR/test.$SRC-$TGT.$TGT | perl ${clean_script_dir}/normalize-punctuation.pl -l ${TGT} -threads ${num_cpus} > $TOK_DATA_DIR/test.$SRC-$TGT.$TGT.norm

    #### ------------------------ 2.2. Tokenize ------------------------ #
    cat $TOK_DATA_DIR/test.$SRC-$TGT.$SRC.norm | perl ${tokenize_script_dir}/moses_tokenizer.pl -a -q -l ${SRC} -no-escape -threads ${num_cpus} > $TOK_DATA_DIR/test.$SRC-$TGT.$SRC
    cat $TOK_DATA_DIR/test.$SRC-$TGT.$TGT.norm | perl ${tokenize_script_dir}/moses_tokenizer.pl -a -q -l ${TGT} -no-escape -threads ${num_cpus} > $TOK_DATA_DIR/test.$SRC-$TGT.$TGT

    ### ------------------------- 2.3. Apply BPE ----------------------- #

    spm_encode --model ${SPM_MODEL_DIR} < ${TOK_DATA_DIR}/test.$SRC-$TGT.$SRC > ${SPM_DATA_DIR}/test.$SRC-$TGT.$SRC
    spm_encode --model ${SPM_MODEL_DIR} < ${TOK_DATA_DIR}/test.$SRC-$TGT.$TGT > ${SPM_DATA_DIR}/test.$SRC-$TGT.$TGT


    fairseq-preprocess \
            --source-lang ${SRC} \
            --target-lang ${TGT} \
            --srcdict ${SPM_VOCAB_DIR} \
            --tgtdict ${SPM_VOCAB_DIR} \
            --testpref ${SPM_DATA_DIR}/test.$SRC-$TGT \
            --destdir ${FAIRSEQ_DATA_DIR} \
            --workers ${num_cpus}

done

find ${TOK_DATA_DIR} -type f -name "*.norm" | xargs rm
echo "Done en-centric"


# ------------------------ 3. Normalize & Apply BPE & Fairseq Zero-Shot ------------------------ #

SPM_DATA_EN_DIR=EC40-FLORES-test/spm/test_en_data

RAW_DATA_DIR=EC40-FLORES-test/raw/test_zs_data 
TOK_DATA_DIR=EC40-FLORES-test/tok/test_zs_data 
SPM_DATA_DIR=EC40-FLORES-test/spm/test_zs_data
FAIRSEQ_DATA_DIR=EC40-FLORES-test/fairseq/test_zs_data

mkdir -p $TOK_DATA_DIR
mkdir -p $SPM_DATA_DIR
mkdir -p $FAIRSEQ_DATA_DIR

SPM_MODEL_DIR=$SPMDIR/spm_64k.model
SPM_VOCAB_DIR=$SPMDIR/fairseq_dict.txt

AR=('de' 'nl' 'sv' 'da' 'is' 'af' 'lb' 'no' 'fr' 'es' 'it' 'pt' 'ro' 'oc' 'ast' 'ca' 'ru' 'cs' 'pl' 'bg' 'uk' 'sr' 'be' 'bs' 'hi' 'bn' 'kn' 'mr' 'sd' 'gu' 'ne' 'ur' 'ar' 'he' 'ha' 'mt' 'ti' 'am' 'kab' 'so')
for i in "${!AR[@]}"; do
    ((j=${i}+1))
    while [ $j -lt ${#AR[@]} ]
    do
        SRC=${AR[i]}
        TGT=${AR[j]}
    
        ### ------------------------ copy spm from en-centric (faster) ------------------------ #

        cat $SPM_DATA_EN_DIR/test.en-$SRC.$SRC > $SPM_DATA_DIR/test.$SRC-$TGT.$SRC
        cat $SPM_DATA_EN_DIR/test.en-$TGT.$TGT > $SPM_DATA_DIR/test.$SRC-$TGT.$TGT

        fairseq-preprocess \
                --source-lang ${SRC} \
                --target-lang ${TGT} \
                --srcdict ${SPM_VOCAB_DIR} \
                --tgtdict ${SPM_VOCAB_DIR} \
                --testpref ${SPM_DATA_DIR}/test.$SRC-$TGT \
                --destdir ${FAIRSEQ_DATA_DIR} \
                --workers ${num_cpus}
        ((j=j+1))
    done
done

find ${TOK_DATA_DIR} -type f -name "*.norm" | xargs rm

mv EC40-FLORES-test/fairseq EC40-FLORES-test-data-bin
rm -R EC40-FLORES-test #keep if you need non-fairseq data

echo "Done zs!"