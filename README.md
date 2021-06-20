# CoMAE

Codes and data for the ACL 2021-Findings paper: **[CoMAE: A Multi-factor Hierarchical Framework for Empathetic Response Generation](https://arxiv.org/abs/2105.08316)**

If you have any problem or suggestion, feel free to contact me: chujiezhengchn@gmail.com

If you use our codes or your research is related to our paper, please kindly cite our paper:

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

## Data

You can download our propossed data [here](https://drive.google.com/drive/folders/1QYRgQb7X-kK6tdM_CwqN8TOhnQJF7m34?usp=sharing). However, the released data is a bit different from the used data in our paper.

- Data size. We found that a RoBERTa classifier may suffer from the unbalanced labels. Hence, for all the factors, we instead use BERT as classifiers. As a result, the filtered data based on CM have a larger size than that reported in our paper
- Taxonomies of DA and EM. We modified the adopted taxonomies of both DA and EM (please refer to the json files in this repo) because:
  - For DA, we found that suggestion is not categorized as a dialog act of expressed empathy (see [the paper of CM](https://arxiv.org/abs/2009.08441)). To keep consistent with the CM paper, we merged suggestion with others
  - For EM, we modified the taxonomies to reduce the overlaps between different emotions
  - Nevertheless, we think you can also modify the taxonomies as needed, and then automatically annotate the utterances

### Performance of BERT-based classifiers

| Classifiers | # classes | Acc  | F1-macro |
| ----------- | --------- | ---- | -------- |
| CM-ER       | 2         | 80.5 | 76.9     |
| CM-IP       | 2         | 84.7 | 84.7     |
| CM-EX       | 2         | 96.8 | 93.6     |
| DA          | 8         | 91.4 | 85.9     |
| EM          | 9         | 65.8 | 62.8     |

### Data Size

| Train  | Valid | Test-Happy | Test-Offmychest |
| ------ | ----- | ---------- | --------------- |
| 154001 | 19940 | 13337      | 7827            |

## Model Implementation

The model implementation is integrated in the repo [chujiezheng/UniModel (github.com)](https://github.com/chujiezheng/UniModel).

