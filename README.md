# Gaussian Multi-head Attention for Simultaneous Machine Translation

Source code for our ACL 2022 paper "Gaussian Multi-head Attention for Simultaneous Machine Translation" (PDF)

Our method is implemented based on the open-source toolkit [Fairseq](https://github.com/pytorch/fairseq).

Core code of Gaussian Multi-head Attention is in [fairseq/modules/gaussian_multihead_attention.py](https://github.com/ictnlp/GMA/blob/main/fairseq/modules/gaussian_multihead_attention.py)



## Requirements and Installation

- Python version = 3.6

- [PyTorch](http://pytorch.org/) version = 1.7

- Install fairseq:

  ```bash
  git clone https://github.com/ictnlp/GMA.git
  cd GMA
  pip install --editable ./
  ```

    

## Quick Start

### Data Pre-processing

We use the data of IWSLT15 English-Vietnamese (download [here](https://nlp.stanford.edu/projects/nmt/)) and WMT15 German-English (download [here](https://www.statmt.org/wmt15/)).

For WMT15 German-English, we tokenize the corpus via [mosesdecoder/scripts/tokenizer/normalize-punctuation.perl](https://github.com/moses-smt/mosesdecoder) and apply BPE with 32K merge operations via [subword_nmt/apply_bpe.py](https://github.com/rsennrich/subword-nmt).

Then, we process the data into the fairseq format, adding `--joined-dictionary` for WMT15 German-English:

```bash
src=SOURCE_LANGUAGE
tgt=TARGET_LANGUAGE
train_data=PATH_TO_TRAIN_DATA
vaild_data=PATH_TO_VALID_DATA
test_data=PATH_TO_TEST_DATA
data=PATH_TO_DATA

# add --joined-dictionary for WMT15 German-English
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${train_data} --validpref ${vaild_data} \
    --testpref ${test_data}\
    --destdir ${data} \
    --workers 20
```

### Training

Train the GMA with the following command:

- delta is the relaxation offset to provide a controllable trade-off between translation quality and latency in practice, and we suggest set delta=1.0.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
data=PATH_TO_DATA
modelfile=PATH_TO_SAVE_MODEL
delta=SET_DELTA

python train.py --ddp-backend=no_c10d ${data} --arch transformer --share-all-embeddings \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --criterion label_smoothed_cross_entropy \
 --label-smoothing 0.1 \
 --left-pad-source False \
 --delta ${delta} \
 --save-dir ${modelfile} \
 --max-tokens 4096 --update-freq 2
```

### Inference

Evaluate the model with the following command:

```bash
export CUDA_VISIBLE_DEVICES=0
data=PATH_TO_DATA
modelfile=PATH_TO_SAVE_MODEL
ref_dir=PATH_TO_REFERENCE

# average last 5 checkpoints
python scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt 

# generate translation
python generate.py ${data} --path $modelfile/average-model.pt --batch-size 250 --beam 1 --left-pad-source False --remove-bpe > pred.out

grep ^H pred.out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.translation
multi-bleu.perl -lc ${ref_dir} < pred.translation
```



## Our Results

The numerical results on IWSLT15 English-to-Vietnamese with Transformer-Small:

| delta |  CW  |  AP  |  AL  | DAL  | BLEU  |
| :---: | :--: | :--: | :--: | :--: | ----- |
|  0.9  | 1.20 | 0.65 | 3.05 | 4.08 | 27.95 |
|  1.0  | 1.27 | 0.68 | 4.01 | 4.77 | 28.20 |
|  2.0  | 1.49 | 0.74 | 5.47 | 6.37 | 28.44 |
|  2.2  | 1.60 | 0.77 | 6.04 | 6.96 | 28.56 |
|  2.5  | 1.74 | 0.78 | 6.55 | 7.55 | 28.72 |

The numerical results on WMT15 German-to-English with Transformer-Base:

| delta |  CW  |  AP  |  AL   |  DAL  | BLEU  |
| :---: | :--: | :--: | :---: | :---: | ----- |
|  0.9  | 1.33 | 0.64 | 3.87  | 4.61  | 28.12 |
|  1.0  | 1.49 | 0.67 | 4.66  | 5.56  | 28.50 |
|  2.0  | 1.85 | 0.72 | 5.79  | 7.75  | 28.71 |
|  2.2  | 2.01 | 0.73 | 6.13  | 8.43  | 29.23 |
|  2.4  | 5.89 | 0.96 | 14.05 | 25.76 | 31.31 |

The numerical results on WMT15 German-to-English with Transformer-Big:

| delta |  CW  |  AP  |  AL   |  DAL  | BLEU  |
| :---: | :--: | :--: | :---: | :---: | ----- |
|  1.0  | 1.54 | 0.68 | 4.60  | 5.89  | 30.20 |
|  2.0  | 1.98 | 0.74 | 6.34  | 8.18  | 30.64 |
|  2.2  | 2.13 | 0.75 | 6.86  | 8.91  | 31.33 |
|  2.4  | 2.28 | 0.76 | 7.28  | 9.59  | 31.62 |
|  2.5  | 3.10 | 0.88 | 12.06 | 20.43 | 31.91 |



## Citation

In this repository is useful for you, please cite as:

```
@inproceedings{GMA,
	title = {Gaussian Multi-head Attention for Simultaneous Machine Translation},
	author = {Shaolei Zhang and Yang Feng},
	booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
	year = {2022},
}
```

