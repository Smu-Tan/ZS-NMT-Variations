import os
import shutil
import argparse

lang_tok_map = {
                
                # Germanic 
                "de": "deu",
                "nl": "nld",
                "sv": "swe",
                "da": "dan",
                "is": "isl",
                "af": "afr",
                "lb": "ltz",
                "no": "nob",

                # Romance
                "fr": "fra",
                "es": "spa",
                "it": "ita",
                "pt": "por",
                "ro": "ron",
                #"oc": occitan is not available
                #"ast": Asturian is not available
                "ca": "cat",

                # Slavic
                "ru": "rus",
                "cs": "ces",
                "pl": "pol",
                "bg": "bul",
                "uk": "ukr",
                "sr": "srp-Latn",
                "be": "bel",
                "bs": "bos",

                # Indo-Aryan
                "hi": "hin",
                "bn": "ben",
                "kn": "kan",
                "mr": "mar",
                "sd": "snd-Arab",
                "gu": "guj",
                "ne": "nep",
                "ur": "urd",

                #Afro-asiatic
                "ar": "arb",
                "he": "heb",
                "ha": "hau",
                "mt": "mlt",
                "ti": "tir",
                "am": "amh",
                "so": "som",
                #"kab": Kabyle is not available

                #en
                "en": "eng-US"
                }

def create_en_lang_pairs(lang_list):
    lang_pairs = []
    for i in lang_list:
        lang_pairs.append("{}-{}".format('en', i))
        #lang_pairs.append("{}-{}".format(i, 'en'))
    print(lang_pairs)

    return lang_pairs


def main(args):

    ntrex_dir = args.ntrex_dir
    output_dir = args.output_dir

    langs = "en,de,nl,sv,da,is,af,lb,no,fr,es,it,pt,ro,ca,ru,cs,pl,bg,uk,sr,be,bs,hi,bn,kn,mr,sd,gu,ne,ur,ar,he,ha,mt,ti,am,so"

    en_lang_list = langs.split(',')
    en_lang_list.remove("en")
    en_lang_pairs = create_en_lang_pairs(en_lang_list)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for lang_pair in en_lang_pairs:
        src_tok, tgt_tok = lang_pair.split("-")
        src_flores_file = os.path.join(ntrex_dir,'newstest2019-ref.' + lang_tok_map[src_tok] + '.txt')
        tgt_flores_file = os.path.join(ntrex_dir, 'newstest2019-ref.' + lang_tok_map[tgt_tok] + '.txt')
        src_fairseq_file = os.path.join(output_dir, "test.{}-{}.{}".format(src_tok, tgt_tok, src_tok))
        tgt_fairseq_file = os.path.join(output_dir, "test.{}-{}.{}".format(src_tok, tgt_tok, tgt_tok))
        shutil.copy(src_flores_file, src_fairseq_file)
        shutil.copy(tgt_flores_file, tgt_fairseq_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntrex-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    main(args)