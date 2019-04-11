Installation
===

```bash
sudo pip3 install virtualenv

sudo apt-get install python3-dev

cd fairseq
virtualenv -p /usr/bin/python3.6 venv-fairseq
. venv-fairseq/bin/activate

pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl

pip3 install torchvision

pip install -r requirements.txt

```


Setup Dataset
===


```bash
# Convolutional Sequence to Sequence Learning
# https://github.com/pytorch/fairseq/tree/master/examples/conv_seq2seq
cd examples/translation/
bash prepare-wmt14en2de.sh
cd ../..

TEXT=examples/translation/wmt14_en_de
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt14_en_de --thresholdtgt 0 --thresholdsrc 0

mkdir -p checkpoints/fconv_wmt_en_de

CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/wmt14_en_de \
  --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 50 \
  --arch fconv_wmt_en_de --save-dir checkpoints/fconv_wmt_en_de

```

```bash
# Scaling Neural Machine Translation
# https://github.com/pytorch/fairseq/tree/master/examples/scaling_nmt

TEXT=examples/scaling_nmt/wmt16_en_de_bpe32k

mkdir -p $TEXT

download https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8

tar -xzvf wmt16_en_de.tar.gz -C $TEXT

fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train.tok.clean.bpe.32000 \
  --validpref $TEXT/newstest2013.tok.bpe.32000 \
  --testpref $TEXT/newstest2014.tok.bpe.32000 \
  --destdir data-bin/wmt16_en_de_bpe32k \
  --nwordssrc 32768 --nwordstgt 32768 \
  --joined-dictionary


CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/wmt16_en_de_bpe32k \
  --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.0005 --min-lr 1e-09 \
  --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 3584 \
  --fp16
```


Train
===


```bash


```

It is better to use max number of tokens instead of batch size. See [here](https://github.com/pytorch/fairseq/issues/143)

https://github.com/pytorch/fairseq/blob/master/fairseq/options.py


**Convolutional Sequence to Sequence Learning (FP32)**

cutoff 4000

Memory Requirement

| Max NUM Tokens | Memory  |
|---|---|
| max-tokens=2000 | 8GB |
| max-tokens=4000 | 11GB |
| max-tokens=8000 | 24GB |
| max-tokens=16000 | 48GB |
| max-tokens=32000 | x |

Throughput (words/sec) 

|   | 2060  | 2070  | 2080  |  1080 Ti | 2080 Ti | TitanRTX | Quadro RTX 6000 | V100 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|---|
| max-tokens=2000  | OOM | 4597 | 6317 | 6207 | 7780 | 8498 | 7407 |  | 7507 |
| max-tokens=4000  | OOM | OOM | OOM | 7290 | 9103 | 10351 | 10157 |  | 10307 |
| max-tokens=8000  | OOM | OOM | OOM | OOM | OOM | 11431 | 11073 |  | 10857 |
| max-tokens=16000  | OOM | OOM | OOM | OOM | OOM | OOM | OOM |  | 11014 |
| max-tokens=32000 | OOM | OOM | OOM | OOM | OOM | OOM | OOM |  | OOM |


**Scaling Neural Machine Translation (FP16)**

cutoff 3584

Memory Requirement

| Max NUM Tokens | Memory  |
|---|---|
| max-tokens=2000 | 8GB |
| max-tokens=3584 | 11GB |
| max-tokens=8000 | 24GB |
| max-tokens=16000 | 48GB | 

Throughput (words/sec) 

|   | 2060  | 2070  | 2080  |  1080 Ti | 2080 Ti | TitanRTX | Quadro RTX 6000 | V100 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|---|
| max-tokens=2000  | OOM | 7721 | 9950 | 5256 | 13558 | 16372 | 16200 |  | 16099 |
| max-tokens=3584  | OOM | OOM | OOM | 5870 | 15671 | 19490 | 19596 |  | 19740 |
| max-tokens=8000  | OOM | OOM | OOM | OOM | OOM | 21180 | 20500 |  | 20539  |
| max-tokens=16000 | OOM | OOM | OOM | OOM | OOM | OOM | OOM |  | 22450 |
