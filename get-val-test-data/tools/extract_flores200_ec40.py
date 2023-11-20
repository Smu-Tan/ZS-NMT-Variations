import os
import shutil
import argparse


lang_tok_map = {

                # Germanic 
                "de": "deu_Latn",
                "nl": "nld_Latn",
                "sv": "swe_Latn",
                "da": "dan_Latn",
                "is": "isl_Latn",
                "af": "afr_Latn",
                "lb": "ltz_Latn",
                "no": "nob_Latn",

                # Romance
                "fr": "fra_Latn",
                "es":"spa_Latn",
                "it": "ita_Latn",
                "pt": "por_Latn",
                "ro": "ron_Latn",
                "oc": "oci_Latn",
                "ast": "ast_Latn",
                "ca": "cat_Latn",

                # Slavic
                "ru": "rus_Cyrl",
                "cs": "ces_Latn",
                "pl": "pol_Latn",
                "bg": "bul_Cyrl",
                "uk": "ukr_Cyrl",
                "sr": "srp_Latn",
                "be": "bel_Cyrl",
                "bs": "bos_Latn",

                # Indo-Aryan
                "hi": "hin_Deva",
                "bn": "ben_Beng",
                "kn": "kan_Knda",
                "mr": "mar_Deva",
                "sd": "snd_Arab",
                "gu": "guj_Gujr",
                "ne": "npi_Deva",
                "ur": "urd_Arab",

                #Afro-asiatic
                "ar": "arb_Arab",
                "he": "heb_Hebr",
                "ha": "hau_Latn",
                "mt": "mlt_Latn",
                "ti": "tir_Ethi",
                "am": "amh_Ethi",
                "so": "som_Latn",
                "kab": "kab_Latn",

                #en
                "en": "eng_Latn"
                }


def main(args):

    langs = "en,de,nl,sv,da,is,af,lb,no,fr,es,it,pt,ro,oc,ast,ca,ru,cs,pl,bg,uk,sr,be,bs,hi,bn,kn,mr,sd,gu,ne,ur,ar,he,ha,mt,ti,am,kab,so"
    flores_dir = args.flores_dir 

    ####  Zero-shot 

    def create_zero_shot_lang_pairs(lang_list):
        if len(lang_list) < 3:
            print("At least three languages!")
            raise

        lang_pairs = []
        i = 0
        while i < len(lang_list):
            j = i + 1
            while j < len(lang_list):
                lang_pairs.append("{}-{}".format(lang_list[i], lang_list[j]))
                j += 1
            i += 1
        print(lang_pairs)
        fairseq_lang_pairs = ','.join(lang_pairs)
        print(fairseq_lang_pairs)
        return lang_pairs

    # create zero-shot pairs
    zero_shot_lang_list = langs.split(',')
    zero_shot_lang_list.remove("en")
    zero_shot_lang_pairs = create_zero_shot_lang_pairs(zero_shot_lang_list)

    output_dir = args.output_dir + '/test_zs_data'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    flores200_merged_dir = os.path.join(flores_dir, "merged")

    for lang_pair in zero_shot_lang_pairs:
        src_tok, tgt_tok = lang_pair.split("-")
        src_flores_file = os.path.join(flores200_merged_dir, lang_tok_map[src_tok] + '.flores200')
        tgt_flores_file = os.path.join(flores200_merged_dir, lang_tok_map[tgt_tok] + '.flores200')
        src_fairseq_file = os.path.join(output_dir, "test.{}-{}.{}".format(src_tok, tgt_tok, src_tok))
        tgt_fairseq_file = os.path.join(output_dir, "test.{}-{}.{}".format(src_tok, tgt_tok, tgt_tok))
        shutil.copy(src_flores_file, src_fairseq_file)
        shutil.copy(tgt_flores_file, tgt_fairseq_file)

    # get En-cenreic
    def create_en_lang_pairs(lang_list):
        lang_pairs = []
        for i in lang_list:
            lang_pairs.append("{}-{}".format('en', i))
            #lang_pairs.append("{}-{}".format(i, 'en'))
        print(lang_pairs)

        return lang_pairs

    en_lang_list = langs.split(',')
    en_lang_list.remove("en")
    en_lang_pairs = create_en_lang_pairs(en_lang_list)

    output_en_dir = args.output_dir + '/test_en_data'

    if os.path.exists(output_en_dir):
        shutil.rmtree(output_en_dir)
    os.makedirs(output_en_dir)

    flores200_test_dir = os.path.join(flores_dir, "merged")
    for lang_pair in en_lang_pairs:
        src_tok, tgt_tok = lang_pair.split("-")
        src_flores_file = os.path.join(flores200_merged_dir, lang_tok_map[src_tok] + '.flores200')
        tgt_flores_file = os.path.join(flores200_merged_dir, lang_tok_map[tgt_tok] + '.flores200')
        src_fairseq_file = os.path.join(output_en_dir, "test.{}-{}.{}".format(src_tok, tgt_tok, src_tok))
        tgt_fairseq_file = os.path.join(output_en_dir, "test.{}-{}.{}".format(src_tok, tgt_tok, tgt_tok))
        shutil.copy(src_flores_file, src_fairseq_file)
        shutil.copy(tgt_flores_file, tgt_fairseq_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flores-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    main(args)