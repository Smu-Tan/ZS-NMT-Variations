# Towards a Better Understanding of Variations in Zero-Shot Neural Machine Translation Performance, EMNLP 2023

The repository of the EMNLP 2023 paper "Towards a Better Understanding of Variations in Zero-Shot Neural Machine Translation Performance", see [preprint](https://arxiv.org/abs/2310.10385).

## EC40

| Resource (size) | Germanic | Romance | Slavic | Indo-Aryan |
| --- | --- | --- | --- | --- |
| High (5M) | de, nl | fr, es | ru, cs | hi, bn | ar, he |
| Medium (1M) | sv, da | it, pt | pl, bg | kn, mr | mt, ha |
| Low (100k) | af, lb | ro, oc | uk, sr | sd, gu | ti, am |
| Extremely-Low (50k) | no, is | ast, ca | be, bs | ne, ur | kab, so |

EC40 is a Multilingual Neural Machine Translation (MNMT) Training Dataset intended to better understand and study MNMT and Zero-Shot NMT. It contains 66 Million English-Centric Sentences covering 40 Languages (excluding English) across 5 Language Families, sampled from OPUS Corpus. 

**Features:**
* _Wide Resource Spectrum:_
   * ranging from High(5M) to Medium(1M), Low(100K), and extremely-Low(50K) resources.
* _Linguistic Diversity:_
   * Each language family is represented at every resource level with two languages, highlighting a balanced and inclusive sampling approach.
* _As a Benchmark:_
   * In total, there are 80 English-centric directions for training and 1,640 directions (including all supervised and ZS directions) for evaluation. Therefore, the EC40 dataset also serves as a benchmark to study multilingual and zero-shot MT.
* _Multi-parallel Utilization:_
   * We make use of Ntrex-128 and Flores-200 as our validation and test datasets, respectively, because of their unique multiparallel characteristics, allowing for further analyses.
   
## Download and Use EC40 as a Benchmark

**We highly recommend you use EC40 in this way unless you want to change the SPM dictionary.**

Please install cyrtranslit first by `pip install cyrtranslit`, which will be used to build test set.

Clone this repo by `git clone https://github.com/Smu-Tan/ZS-NMT-Variations.git`, then run scripts under this directory.

* Step 1: [Download EC40 Fairseq data-bin](https://drive.google.com/file/d/1Miqeg94qzBr3i_JcoyNOAPhlRzk9flG2/view?usp=drive_link). We provide the Fairseq Binarized EC40 Training set for easy training. If you want to use the EC40 as a benchmark (with its original SentencePiece dictionary), then you should download this. Note: the data-bin is sharded to avoid high RAM consumption.

* Step 2: [Prepare Validation and test set](https://github.com/Smu-Tan/ZS-NMT-Variations/tree/main/get-val-test-data/get_fairseq_format_scripts). We provide the Scripts building the validation and test set using Ntrex-128 and Flores-200. If you want to use the EC40 as a benchmark (with its original SentencePiece dictionary), then you should follow this. Note: we merged the Flores-200 _dev_ and _dev-test_ as the final test set.

* Step 3: [copy val set to _fairseq-data-bin-sharded_](https://github.com/Smu-Tan/ZS-NMT-Variations/tree/main/get-val-test-data/get_fairseq_format_scripts). This step is to make sure the val set is contained in the training set (fairseq training fashion).

* [download trained SPM Dictionary and Model](https://drive.google.com/drive/folders/1tsZzQraZ7nXTyYjUCaM_JVWUnKRe_aAa?usp=drive_link). You can download our trained SentencePiece Dictionary and Model (`it is also contained under ZS-NMT-Variations/get-val-test-data/spm_dict`)

* [Training scripts](https://github.com/Smu-Tan/ZS-NMT-Variations/tree/main/train-eval-baseline-scripts). Scripts for training baseline models on EC40.

* [Evaluation scripts](https://github.com/Smu-Tan/ZS-NMT-Variations/tree/main/train-eval-baseline-scripts). Scripts for evaluating baseline models on both Supervised and Zero-Shot directions.

* [Baseline Model Checkpoints](https://drive.google.com/drive/folders/1H9PU05mriTHWCWTFXsOM7KYlXpqZte7T?usp=drive_link). We also provide Checkpoints of baseline models.


## Download "Plain" EC40 Dataset

Please install cyrtranslit first by `pip install cyrtranslit`, which will be used to build test set.

Clone this repo by `git clone https://github.com/Smu-Tan/ZS-NMT-Variations.git`, then run scripts under this directory.

To use "Plain" EC40, we provide the Simplified Procedure below:
1. Download Plain EC40 Dataset and prepare the val & test sets.
2. Train your own SPM dict and model. (Otherwise, Go to the section `Use EC40 as a Benchmark`)
3. Build the Sharded Dataset

* [Download EC40 Dataset (Plain)](https://drive.google.com/drive/folders/1nZsDnj3mNKynk2D46frnLfmR9qTFzVM9?usp=drive_link). Here "Plain" means it is not processed by the BPE, all data are in txt format. EC40 is open-to-use, we carefully pre-processed it. thus, no need to run additional preprocessing commands like deduplication, Moses normalization, etc.

* [Prepare Validation and test set (Plain)](https://github.com/Smu-Tan/ZS-NMT-Variations/tree/main/get-val-test-data/get_plain_scripts). We provide the Scripts building the "Plain" validation and test set using Ntrex-128 and Flores-200. Note: we merged the Flores-200 _dev_ and _dev-test_ as the final test set.

* [Script of Building Sharded Dataset](https://drive.google.com/file/d/1FAJEHcv8rM06iKF4zbuk2IYs2i6Tyz5A/view?usp=drive_link). The script template of how to build the sharded dataset if you use the "Plain" dataset. You do not have to follow this step if you want to use Huggingface or other tools than Fairseq.


## Citation

Please cite both our paper (tan2023towards) and OPUS (tiedemann2012parallel) when you only use the EC40 **training** dataset.

```
@article{tan2023towards,
  title={Towards a Better Understanding of Variations in Zero-Shot Neural Machine Translation Performance},
  author={Tan, Shaomu and Monz, Christof},
  journal={arXiv preprint arXiv:2310.10385},
  year={2023}
}
```
```
@inproceedings{tiedemann2012parallel,
  title={Parallel data, tools and interfaces in OPUS.},
  author={Tiedemann, J{\"o}rg},
  booktitle={Lrec},
  volume={2012},
  pages={2214--2218},
  year={2012},
  organization={Citeseer}
}
```

Please also cite Ntrex-128 and Flores-200 if you use the same validation and test dataset.

```
@inproceedings{federmann2022ntrex,
  title={NTREX-128--news test references for MT evaluation of 128 languages},
  author={Federmann, Christian and Kocmi, Tom and Xin, Ying},
  booktitle={Proceedings of the First Workshop on Scaling Up Multilingual Evaluation},
  pages={21--24},
  year={2022}
}
```
```
@article{costa2022no,
  title={No language left behind: Scaling human-centered machine translation},
  author={Costa-juss{\`a}, Marta R and Cross, James and {\c{C}}elebi, Onur and Elbayad, Maha and Heafield, Kenneth and Heffernan, Kevin and Kalbassi, Elahe and Lam, Janice and Licht, Daniel and Maillard, Jean and others},
  journal={arXiv preprint arXiv:2207.04672},
  year={2022}
}
```

## Acknowledgements
* EC40 is sampled from OPUS Corpus. We thank Jörg Tiedemann and other researchers who contributed to the OPUS. 
* We thank researchers who contributed to the wonderful Ntrex-128 and Flores-200. 

