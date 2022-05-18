This repository contains the code and embeddings from our Findings of ACL 2022 publication "[Combining static and contextual representations](https://aclanthology.org/2022.findings-acl.182/)".
It includes:

1. `bash-scripts` with the calling parameters for the continued pre-training.
2. `continued_pretraining` code along with unit tests.
3. `extraction_data_preproc` code to create the data for X2S-M/X2S-MA.

# Installation

The `requirements.txt` file lists the top-level dependencies you need to install to use each part of the project.
You may want to use separate Python environments for the two parts.

# Usage

We provide the calling scripts used for continued pre-training in `bash-scripts`.
These are the exact settings used for the models in the paper, except for the directory path placeholders.

For the static embedding extraction code installation and usage, refer to [the X2Static repository](https://github.com/epfml/X2Static). To apply this to XLM-R, we only made minimal changes to `learn_from_roberta_ver2.py`, i.e., using `AutoModel` and `AutoTokenizer` instead of `RobertaModel` and `RobertaTokenizer` when loading the contextual model.

# Embeddings

You can download the final, aligned embeddings [X2S-MA](https://www.cis.uni-muenchen.de/~haemmerl/resources/X2S_MA.tgz) from my website:

```
wget https://www.cis.uni-muenchen.de/~haemmerl/resources/X2S_MA.tgz
tar -xzf X2S_MA.tgz
```

# Citation

If you find our code or resources helpful, please cite our paper:

```
@inproceedings{hammerl-etal-2022-combining,
    title = "Combining Static and Contextualised Multilingual Embeddings",
    author = {H{\"a}mmerl, Katharina  and
      Libovick{\'y}, Jind{\v{r}}ich  and
      Fraser, Alexander},
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.182",
    pages = "2316--2329",
}
```
