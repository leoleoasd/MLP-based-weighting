<div align="center">    
<h1>Contextual embedding and model weighting by fusing domain knowledge on Biomedical Question Answering </h1>
<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  

ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)
-->

<!--  
Conference   
-->   
</div>
 
## Description   
A Contextual Embedding and Model Weighting strategy.

BioBERT is used in AoA Reader to provide biomedical contextual word embedding. Then the result of AoA Reader is weighted with the result of SciBERT to achieve a better performance.

We achieve the accuracy of 88.00% on the BIOMRC LITE dataset, outperforms state-of-the-art systems by a large margin of 8.03%.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/leoleoasd/MLP-based-weighting

# install project   
cd MLP-based-weighting
pip install -e .   
conda env create -f conda-env.yaml
 ```
Download biobert from huggingface.

Next, navigate to the project and run it.
 ```bash
# run aoa reader
python3 -m aoa_reader --batch_size 30 --precision=16 --name biobert_final --bert_dir data/bert_huggingface --gpus=,1 --occ_agg=sum --tok_agg=sum
# run scibert
python3 bert/biomrc_origional.py
# get score for single model
python3 -m aoa_reader --batch_size 30 --precision=16 ---bert_dir 'data/bert_huggingface' --not_train --not_test --test_ckpt_path aoa_reader.ckpt --predict
# get score for single model in scibert:
# change `doEval` and `doTrain` in bert/biomrc_original.py, then
python3 bert/biomrc_origional.py
# run MLP-based weighting
python3 -m weight --batch_size 1000 --aoa_filename ./predictions.pt --bert_filename ./scibert_predict.pt
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from aoa_reader.biomrc_data import BioMRDataModule
from aoa_reader.model import AOAReader_Model
from pytorch_lightning import Trainer
```

## License
The `aoa_reader` and `weight` part of this project is licensed under the GPL v3 public license.

The `bert_original.py` is extracted from [PetrosStav/BioMRC_code](https://github.com/PetrosStav/BioMRC_code).

### Citation   

This paper isn't published yet. Will update info after publication.
<!--```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```    -->

### Acknowledgements

Thanks to [the code released by pappas](https://github.com/PetrosStav/BioMRC_code).

Thanks to [the pytorch lightning team](https://www.pytorchlightning.ai/) for creating an easy-to-use library and repository template.