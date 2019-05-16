# Cooperative Learning of Disjoint Syntax and Semantics
This code was used to obtain the results described in the paper:

**[Cooperative Learning of Disjoint Syntax and Semantics](https://arxiv.org/abs/1902.09393)**
    <br>
    <a href='https://serhii-havrylov.github.io'>Serhii Havrylov</a>,
    <a href='http://germank.github.io'>Germán Kruszewski</a>,
    <a href='https://research.fb.com/people/joulin-armand'>Armand Joulin</a>
    <br>
    Presented at [NAACL2019](https://naacl2019.org/program/accepted/)

## ListOps
1. Download ListOps dataset. URLs of the original dataset and an extrapolation test set can be found in `data/listops/external/urls.txt` file.
2. Run `python listops/data_preprocessing/split.py` to split the dataset into the train, the valid and the test sets. 
   Make sure that you have downloaded the dataset and the files are present in the `data/listops/external` folder.
3. Build vocabulary using `python listops/data_preprocessing/build_vocab.py`.
4. Run `python listops/ppo/train_ppo_model.py` or `python listops/reinforce/train_reinforce_model.py` to train the model with PPO or REINFORCE estimators.

## SST
1. Run `python sst/ppo/train_ppo_model.py` to train the model using SST-2 or SST-5 datasets.

## NLI
1. Download SNLI and MultiNLI datasets and extract corresponding archives to `data/nli` folder. URLs can be found in `data/nli/external/urls.txt` file.
2. Run `python nli/data_preprocessing/preprocess.py` to preprocess dataset and generate vocabulary files.
3. Run `python nli/ppo/train_ppo_model.py` to train the model using SNLI or MultiNLI datasets.

The code is tested with Python 3.6.3 and PyTorch 1.0.1.
 
## License

Latent-TreeLSTM is MIT licensed. See the LICENSE file for details.
