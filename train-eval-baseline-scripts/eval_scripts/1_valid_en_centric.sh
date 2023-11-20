#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --job-name=eval
#SBATCH --exclude=ilps-cn[002-008],ilps-cn[101-114]
#SBATCH --time=20-00:00:00
#SBATCH --mem=32G

pwd
conda info --envs
source /home/stan1/anaconda3/bin/activate fairseq
cd /ivi/ilps/personal/stan1/zero_shot_project/EC40/


PAIRS=('en-de' 'en-nl' 'en-sv' 'en-da' 'en-is' 'en-af' 'en-lb' 'en-no' 'en-fr' 'en-es' 'en-it' 'en-pt' 'en-ro' 
       'en-oc' 'en-ast' 'en-ca' 'en-ru' 'en-cs' 'en-pl' 'en-bg' 'en-uk' 'en-sr' 'en-be' 'en-bs' 'en-hi' 'en-bn' 
       'en-kn' 'en-mr' 'en-sd' 'en-gu' 'en-ne' 'en-ur' 'en-ar' 'en-he' 'en-ha' 'en-mt' 'en-ti' 'en-am' 'en-kab' 'en-so')


RESULT_DIR=checkpoints/m2m_base_monitor_shard_new_large/results-flores200/en-centric

mkdir checkpoints/m2m_base_monitor_shard_new_large/results-flores200
mkdir $RESULT_DIR

for i in "${!PAIRS[@]}"; do
    PAIR=${PAIRS[i]}
    SRC=${PAIR%-*}
    TGT=${PAIR#*-}
    
    ### forward translation
    fairseq-generate flores-data-bin-en \
        --langs en,de,nl,sv,da,is,af,lb,no,fr,es,it,pt,ro,oc,ast,ca,ru,cs,pl,bg,uk,sr,be,bs,hi,bn,kn,mr,sd,gu,ne,ur,ar,he,ha,mt,ti,am,kab,so \
        --lang-pairs $SRC-$TGT \
        --source-lang $SRC \
        --target-lang $TGT \
        --max-tokens 10240 \
        --arch transformer_vaswani_wmt_en_de_big \
        --task translation_multi_simple_epoch \
        --fp16 \
        --encoder-langtok tgt \
        --remove-bpe 'sentencepiece' \
        --tokenizer space \
        --path checkpoints/mt_large/mt_large.pt \
        --skip-invalid-size-inputs-valid-test \
        --gen-subset test \
        --seed 1234 \
        --results-path checkpoints/mt_large/results-flores200/en-centric/${SRC}-${TGT}

    # extract results
    grep ^H ${RESULT_DIR}/${SRC}-${TGT}/generate-test.txt | LC_ALL=C sort -V | cut -f3- | sacremoses -l ${TGT} detokenize > ${RESULT_DIR}/${SRC}-${TGT}/test-sys.txt
    grep ^T ${RESULT_DIR}/${SRC}-${TGT}/generate-test.txt | LC_ALL=C sort -V | cut -f2- | sacremoses -l ${TGT} detokenize > ${RESULT_DIR}/${SRC}-${TGT}/test-ref.txt
    grep ^S ${RESULT_DIR}/${SRC}-${TGT}/generate-test.txt | LC_ALL=C sort -V | awk -F'\t' '{ sub(/.*__[a-z]+__/, ""); print }' | sacremoses -l ${SRC} detokenize > ${RESULT_DIR}/${SRC}-${TGT}/test-src.txt

    # sacrebleu
    sacrebleu ${RESULT_DIR}/${SRC}-${TGT}/test-ref.txt -i ${RESULT_DIR}/${SRC}-${TGT}/test-sys.txt -l ${SRC}-${TGT} > ${RESULT_DIR}/${SRC}-${TGT}/test_bleu.txt

    # chrf++
    sacrebleu ${RESULT_DIR}/${SRC}-${TGT}/test-ref.txt -i ${RESULT_DIR}/${SRC}-${TGT}/test-sys.txt -l ${SRC}-${TGT} -m chrf --chrf-word-order 2 > ${RESULT_DIR}/${SRC}-${TGT}/test_chrfpp.txt

    # comet
    SOURCE_SENT=${RESULT_DIR}/${SRC}-${TGT}/test-src.txt
    HYPOTHESIS=${RESULT_DIR}/${SRC}-${TGT}/test-sys.txt
    REFERENCE=${RESULT_DIR}/${SRC}-${TGT}/test-ref.txt
    comet-score -s ${SOURCE_SENT} -t ${HYPOTHESIS} -r ${REFERENCE} --quiet --only_system > ${RESULT_DIR}/${SRC}-${TGT}/test_comet.txt
    
    # SpBleu
    # tokenize with SPM
    python eval_scripts/spm_encode.py \
        --model flores200_sacrebleu_tokenizer_spm.model \
        --output_format=piece \
        --inputs=${HYPOTHESIS} \
        --outputs=${RESULT_DIR}/${SRC}-${TGT}/test-sp-sys.txt
    HYPOTHESIS=${RESULT_DIR}/${SRC}-${TGT}/test-sp-sys.txt
    python eval_scripts/spm_encode.py \
        --model flores200_sacrebleu_tokenizer_spm.model \
        --output_format=piece \
        --inputs=${REFERENCE} \
        --outputs=${RESULT_DIR}/${SRC}-${TGT}/test-sp-ref.txt
    REFERENCE=${RESULT_DIR}/${SRC}-${TGT}/test-sp-ref.txt
    # calculate spbleu
    cat ${HYPOTHESIS} | sacrebleu ${REFERENCE} > ${RESULT_DIR}/${SRC}-${TGT}/test_spbleu.txt


    ### backward translation
    fairseq-generate flores-data-bin-en \
        --langs en,de,nl,sv,da,is,af,lb,no,fr,es,it,pt,ro,oc,ast,ca,ru,cs,pl,bg,uk,sr,be,bs,hi,bn,kn,mr,sd,gu,ne,ur,ar,he,ha,mt,ti,am,kab,so \
        --lang-pairs $TGT-$SRC \
        --source-lang $TGT \
        --target-lang $SRC \
        --max-tokens 10240 \
        --arch transformer_vaswani_wmt_en_de_big \
        --task translation_multi_simple_epoch \
        --fp16 \
        --encoder-langtok tgt \
        --remove-bpe 'sentencepiece' \
        --tokenizer space \
        --path checkpoints/mt_large/mt_large.pt \
        --skip-invalid-size-inputs-valid-test \
        --gen-subset test \
        --seed 1234 \
        --results-path checkpoints/mt_large/results-flores200/en-centric/${TGT}-${SRC} \

    # extract results
    grep ^H ${RESULT_DIR}/${TGT}-${SRC}/generate-test.txt | LC_ALL=C sort -V | cut -f3- | sacremoses -l ${TGT} detokenize > ${RESULT_DIR}/${TGT}-${SRC}/test-sys.txt
    grep ^T ${RESULT_DIR}/${TGT}-${SRC}/generate-test.txt | LC_ALL=C sort -V | cut -f2- | sacremoses -l ${TGT} detokenize > ${RESULT_DIR}/${TGT}-${SRC}/test-ref.txt
    grep ^S ${RESULT_DIR}/${TGT}-${SRC}/generate-test.txt | LC_ALL=C sort -V | awk -F'\t' '{ sub(/.*__[a-z]+__/, ""); print }' | sacremoses -l ${TGT} detokenize > ${RESULT_DIR}/${TGT}-${SRC}/test-src.txt
    
    # sacrebleu
    sacrebleu ${RESULT_DIR}/${TGT}-${SRC}/test-ref.txt -i ${RESULT_DIR}/${TGT}-${SRC}/test-sys.txt -l ${TGT}-${SRC} > ${RESULT_DIR}/${TGT}-${SRC}/test_bleu.txt

    # chrf++
    sacrebleu ${RESULT_DIR}/${TGT}-${SRC}/test-ref.txt -i ${RESULT_DIR}/${TGT}-${SRC}/test-sys.txt -l ${TGT}-${SRC} -m chrf --chrf-word-order 2 > ${RESULT_DIR}/${TGT}-${SRC}/test_chrfpp.txt
    
    # comet
    SOURCE_SENT=${RESULT_DIR}/${TGT}-${SRC}/test-src.txt
    HYPOTHESIS=${RESULT_DIR}/${TGT}-${SRC}/test-sys.txt
    REFERENCE=${RESULT_DIR}/${TGT}-${SRC}/test-ref.txt
    comet-score -s ${SOURCE_SENT} -t ${HYPOTHESIS} -r ${REFERENCE} --quiet --only_system > ${RESULT_DIR}/${TGT}-${SRC}/test_comet.txt
    
    # SpBleu
    # tokenize with SPM
    python eval_scripts/spm_encode.py \
        --model eval_scripts/flores200_sacrebleu_tokenizer_spm.model \
        --output_format=piece \
        --inputs=${HYPOTHESIS} \
        --outputs=${RESULT_DIR}/${TGT}-${SRC}/test-sp-sys.txt
    HYPOTHESIS=${RESULT_DIR}/${TGT}-${SRC}/test-sp-sys.txt
    python eval_scripts/spm_encode.py \
        --model eval_scripts/flores200_sacrebleu_tokenizer_spm.model \
        --output_format=piece \
        --inputs=${REFERENCE} \
        --outputs=${RESULT_DIR}/${TGT}-${SRC}/test-sp-ref.txt
    REFERENCE=${RESULT_DIR}/${TGT}-${SRC}/test-sp-ref.txt
    # calculate spbelu
    cat ${HYPOTHESIS} | sacrebleu ${REFERENCE} > ${RESULT_DIR}/${TGT}-${SRC}/test_spbleu.txt


done

