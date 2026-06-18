# MIMIC-III/MIMIC-IV Patient Mortality Prediction w/ EHR and Text Reports

# Requirements
- A CUDA-supported GPU.
- CUDA Toolkit 12.6 or greater.
- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) or pip (Recommended to use uv)

# Installation
1. Clone the repository with:
```sh
git clone https://github.com/ForceLightning/mimic-iii-mortality.git
```
2. Sync dependencies from `pyproject.toml` or `requirements.txt` or `uv.lock`. Note that the CUDA version used here is 12.6, modify as necessary.
```sh
pip install -r requirements.txt
# or
uv add -r requirements.txt
# or
uv sync
```
# Usage
> !NOTE
> Ensure that the environment variable `PYTHONPATH` is set to `./src/`.
> This can be done with:
> ```sh
> export PYTHONPATH="src/"
> ```
> Alternatively, set it up in a `.env` file if using `uv`, and run `uv` commands with the arguments `--env-file .env`.

The main command is as follows:
```sh
TOKENIZERS_PARALLELISM=true uv run --env-file .env -- python -m Main_modality_model ...
```
with the following arguments:
- `model_type` (str/Enum) \[Mandatory\]: `RNN`, `LSTM`, `TRANSFORMER`, `BERT`, `BERT_EMB`, `BERT_EHR`, `BERT_EMB_EHR_TCN`, or `ENN_EHR`.
- `data_type` (str/Enum) \[Mandatory\]: `EHR`, `REPORT`, `EHR_AND_REPORT`, `BCB_EMB`, `BCB_EMB_EHR`, `PHENOTYPE_BCB_EMB`, `PHENOTYPE_BCB_EMB_EHR` or `EVID_EMB_EHR`.
- `data_dir` (str): Directory to train/val/test data.
- `checkpoints_dir` (str): Directory to save model checkpoints to.
- `batch_size` (int): Minibatch size.
- `hidden_size` (int): Hidden layer size for RNN and LSTM.
- `num_layers` (int): Number of hidden layers for RNN and LSTM.
- `num_epoch` (int): Total number of epochs to run.
- `lr` (float): Learning rate.
- `trans_input_dim` (int): Transformer input layer size.
- `trans_n_heads` (int): Transformer num heads.
- `trans_ff_dim` (int): Transformer feed forward layer size.
- `trans_dropout` (float): Transformer dropout rate.
- `trans_num_layers` (int): Number of transformer layers.
- `trans_max_len` (int): Total number of tokens to use before truncation.
- `output_l1_dim` (int): Linear layer 1 size.
- `output_l2_dim` (int): Linear layer 2 size.
- `output_l3_dim` (int): Linear layer 3 size (for RNN, LSTM, and Transformer models only).
- `bert_use_temporal_conv` (bool): Use temporal convolutions over the temporal axis for embeddings.
- `bert_model_str` (str): BERT model definition name from Huggingface.
- `seed_everything` (int): Global seed for RNG.
- `show_auc_plots` (bool): Whether to show the AUROC/AUPRC.
- `show_confusion_matrix` (bool): Whether to show the classification confusion matrix after testing.

## Third party parameters
- `enn_prototype_dim` (int): Number of prototypes $H$ in input layer.
- `enn_n_blocks` (int): Number of blocks in MLP layer.
- `enn_structured_d_hidden` (int): Size of hidden layer for EHR features before classification.
- `enn_notes_d_hidden` (int): Size of hidden layer for textual embedding features before classification.
- `enn_alpha1` (float): Scaling factor of the 1st auxiliary loss.
- `enn_alpha2` (float): Scaling factor of the 2nd auxiliary loss.
- `enn_dropout` (float): Dropout factor for ENN.
- `enn_structured_d_in_ls` (list\[int\]): Sizes of categorical and continuous features in the EHR data.

See [train.sh](./train.sh ) for an example of how the arguments may be used.
