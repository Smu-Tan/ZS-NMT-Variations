# EC40 MNMT Dataset

### EC40 is an English-Centric Multilingual Machine Translation Dataset. It has over 60 Million sentences including 40 Languages across 5 Language Families. 

### Features:
1. We carefully balanced the dataset across resources and languages by strictly maintaining each resource group containing 5 language families and each family consists of 8 representative languages.
2. EC40 covers a wide spectrum of resource availability, ranging from High(5M) to Medium(1M), Low(100K), and extremely-Low(50K) resources.
3. In total, there are 80 English-centric directions for training and 1,640 directions (including all supervised and ZS directions) for evaluation.
4. We make use of Ntrex-128 and Flores-200 as our validation and test set.


-----
## Languages and Family

| Family | Languges | 
| :---         |     :---:      | 
| Germanic   | Geman, Dutch, Swedish, Danish, Afrikaans, Luxembourgish, Norwegian, Icelandic     | 
| Romance    | French, Spanish, Italian, Portuguese, Romanian, Occitan, Asturian, Catalan   | 
| Slavic	 | Russian, Czech, Polish, Bulgarian, Ukrainian, Serbian, Belarusian, Bosnian |
| Indo-Aryan  | Hindi, Bengali, Kannada, Marathi, Sindhi, Gujarati, Nepali, Urdu |
-----

## Dataset Stats

| Resource | Languages | Size |
| --- | --- | --- |
| High | de, nl, fr, es, ru, cs, hi, bn, ar, he | 5M |
| Medium | sv, da, it, pt, pl, bg, kn, mr, mt, ha | 1M |
| Low | af, lb, ro, oc, uk, sr, sd, gu, ti, am | 100k |
| Extremely-Low | no, is, ast, ca, be, bs, ne, ur, kab, so | 50k |

-----
## Build Fairseq dataset (Shard->to avoid RAM OOM)

```
Read toolkit/build_fairseq_sharded_dataset.sh
```
<br>

-----

## Train mTransformer-Large baseline

```
Read toolkit/train-EC40-mTrans-large.sh
```
