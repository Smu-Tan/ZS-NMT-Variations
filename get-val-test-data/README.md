## Prepare val and test set (Fairseq Bin format)

**We highly recommend you use EC40 in this way unless you want to change the SPM dictionary.**

Please install cyrtranslit first by `pip install cyrtranslit`, which will be used to build test set.

Clone this repo by `git clone https://github.com/Smu-Tan/ZS-NMT-Variations.git`, then run scripts under this directory.

* Step 1: [Prepare Validation and test set](https://github.com/Smu-Tan/ZS-NMT-Variations/tree/main/get-val-test-data/get_fairseq_format_scripts). We provide the Scripts building the validation and test set using Ntrex-128 and Flores-200. If you want to use the EC40 as a benchmark (with its original SentencePiece dictionary), then you should follow this. Note: we merged the Flores-200 _dev_ and _dev-test_ as the final test set.

* Step 2: [copy val set to _fairseq-data-bin-sharded_](https://github.com/Smu-Tan/ZS-NMT-Variations/tree/main/get-val-test-data/get_fairseq_format_scripts). This step is to make sure the val set is contained in the training set (fairseq training fashion).

## Prepare val and test set (Plain text format)

Please install cyrtranslit first by `pip install cyrtranslit`, which will be used to build test set.

Clone this repo by `git clone https://github.com/Smu-Tan/ZS-NMT-Variations.git`, then run scripts under this directory.

* [Prepare Validation and test set (Plain)](https://github.com/Smu-Tan/ZS-NMT-Variations/tree/main/get-val-test-data/get_plain_scripts). We provide the Scripts building the "Plain" validation and test set using Ntrex-128 and Flores-200. Note: we merged the Flores-200 _dev_ and _dev-test_ as the final test set.


## Acknowledgements
* We utilize Ntrex-128 and Flores-200 as our validation and test set, and we thank researchers who contributed to these datasets. Please also cite their papers if you use them.
