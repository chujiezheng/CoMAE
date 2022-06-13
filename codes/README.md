# Running Scripts for *CoMAE*

**Chujie Zheng**, Yong Liu, Wei Chen, Yongcai Leng and Minlie Huang. **CoMAE: A Multi-factor Hierarchical Framework for Empathetic Response Generation**. *In Findings of ACL 2021*. [[paper]](https://arxiv.org/abs/2105.08316) [[repo]](https://github.com/chujiezheng/CoMAE)

```bib
@inproceedings{zheng-etal-2021-comae,
    title = "CoMAE: A Multi-factor Hierarchical Framework for Empathetic Response Generation",
    author = "Zheng, Chujie  and
      Liu, Yong  and
      Chen, Wei  and
      Leng, Yongcai  and
      Huang, Minlie",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2021",
    year = "2021",
}
```

## Preparing Enviroment

```bash
conda env create -f env.yml -n UniModel
conda activate UniModel
```

## Downloading Model

You should download the [GPT-2](https://huggingface.co/gpt2) model and replace the fake `pytorch_model.bin` file in `GPT2-small` with the true one.

If you would like to evaluate generated results with Embedding-based similarity, you can download my prepared embedding files from [here](https://drive.google.com/drive/folders/11TwzwDtQoFHynlG0b1uT1MPQz9Jctb66?usp=sharing).

## Preprocessing Training Data

First, put the downloaded data into `_reformat`.

Then, run `bash RUN/comae/prepare.sh` to preprocess the training data.

## Training Your Model

Run `bash RUN/comae/train.sh` to train your model.

## Inference with Your Model

Every time of model training will create a new folder in `DATA/comae/comae.comae`, which is named after the time when the training starts. You should select a checkpoint (it may be based on the PPL of validation), and replace the checkpoint path in `RUN/comae/infer.sh --load_checkpoint` with the path of your selected checkpoint.

Then, run `bash RUN/comae/infer.sh` to do the inference.

**Note:** while we used GPT-2 as the backbone model in our paper, we found that using DialoGPT usually leads to better performance (especially in terms of PPL). However, the optimal decoding parameters for these two pre-trained models are different. Based on my experiences, top-p 0.9 and temperature 0.7 are always good, but the repetition_penalty 1.05 is better for DialoGPT while 1.5 for GPT-2.
