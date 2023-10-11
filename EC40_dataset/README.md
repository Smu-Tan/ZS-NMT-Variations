# EC40 MT Dataset -Shaomu Tan

### EC40 is an English-Centric Multilingual Machine Translation Dataset. It has over 60 Million sentences including 40 Languages across 5 Language Families. 

## DO NOT Change ANYTHING in this folder.

<br>

-----
## Languages and Family

| Family | Languges | 
| :---         |     :---:      | 
| Germanic   | Geman, Dutch, Swedish, Danish, Afrikaans, Luxembourgish, Norwegian, Icelandic     | 
| Romance    | French, Spanish, Italian, Portuguese, Romanian, Occitan, Asturian, Catalan   | 
| Slavic	 | Russian, Czech, Polish, Bulgarian, Ukrainian, Serbian, Belarusian, Bosnian |
| Indo-Aryan  | Hindi, Bengali, Kannada, Marathi, Sindhi, Gujarati, Nepali, Urdu |
|

<br>

-----

## Dataset Stats

| Resource | Languages | Size |
| --- | --- | --- |
| High | de, nl, fr, es, ru, cs, hi, bn, ar, he | 5M |
| Medium | sv, da, it, pt, pl, bg, kn, mr, mt, ha | 1M |
| Low | af, lb, ro, oc, uk, sr, sd, gu, ti, am | 100k |
| Extremely-Low | no, is, ast, ca, be, bs, ne, ur, kab, so | 50k |

<br>

-----
## Build Fairseq dataset (Sharded)

```
Read toolkit/build_fairseq_sharded_dataset.sh
```
<br>

-----

## Train mTransformer-Large baseline

```
Read toolkit/train-EC40-mTrans-large.sh
```