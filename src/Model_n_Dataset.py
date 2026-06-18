# Standard Library
import os
from enum import Enum, auto
from functools import partial
from typing import Literal, override

# Scientific Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

# PyTorch
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

# Huggingface imports
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer


class DatasetType(Enum):
    EHR = auto()
    REPORT = auto()
    EHR_AND_REPORT = auto()
    BCB_EMB = auto()
    BCB_EMB_EHR = auto()
    PHENOTYPE_BCB_EMB = auto()
    PHENOTYPE_BCB_EMB_EHR = auto()
    EVID_EMB_EHR = auto()

    def get_train_test_dir(self) -> tuple[str, str]:
        match self:
            case (
                DatasetType.EHR
                | DatasetType.REPORT
                | DatasetType.EHR_AND_REPORT
                | DatasetType.BCB_EMB_EHR
                | DatasetType.BCB_EMB
                | DatasetType.EVID_EMB_EHR
            ):
                return "train_with_raw_report", "test_with_raw_report"
            case DatasetType.PHENOTYPE_BCB_EMB | DatasetType.PHENOTYPE_BCB_EMB_EHR:
                return (
                    "phenotyping_first_48_hour/train",
                    "phenotyping_first_48_hour/test",
                )

    def get_dataset_class(self):
        match self:
            case DatasetType.EHR:
                return EHRDataset
            case DatasetType.REPORT:
                return ReportDataset
            case DatasetType.EHR_AND_REPORT:
                return EHRAndReportDataset
            case DatasetType.BCB_EMB | DatasetType.PHENOTYPE_BCB_EMB:
                return BioclinicalBERTEmbeddingsDataset
            case DatasetType.BCB_EMB_EHR | DatasetType.PHENOTYPE_BCB_EMB_EHR:
                return EHRAndBioclinicalBERTEmbeddingsDataset
            case DatasetType.EVID_EMB_EHR:
                return EVIDEHRAndBioclinalBERTEmbeddingsDataset

    def get_eval_fn(self):
        match self:
            case (
                DatasetType.EHR
                | DatasetType.REPORT
                | DatasetType.EHR_AND_REPORT
                | DatasetType.BCB_EMB
                | DatasetType.BCB_EMB_EHR
                | DatasetType.EVID_EMB_EHR
            ):
                return (
                    roc_auc_score,
                    average_precision_score,
                    precision_score,
                    recall_score,
                    f1_score,
                )
            case DatasetType.PHENOTYPE_BCB_EMB | DatasetType.PHENOTYPE_BCB_EMB_EHR:
                ret = [
                    roc_auc_score,
                    average_precision_score,
                    precision_score,
                    recall_score,
                    f1_score,
                ]
                return (partial(fn, average="micro") for fn in ret)

    def __str__(self) -> str:
        return str(self.name)


class ModelType(Enum):
    RNN = auto()
    LSTM = auto()
    TRANSFORMER = auto()
    BERT = auto()
    BERT_EMB = auto()
    BERT_EHR = auto()
    BERT_EMB_EHR_TCN = auto()
    ENN_EHR = auto()

    def __str__(self) -> str:
        return str(self.name)


# define and create dataset (using processed EHR from MIMIC III data)
class EHRDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, root_dir: str):
        self.ehr_dir = os.path.join(root_dir, "EHR/")
        label_path = os.path.join(root_dir, "labels.csv")

        self.labels_df = pd.read_csv(label_path, index_col=False)
        self.stay_ids = self.labels_df["stay"].astype(str)
        self.labels = self.labels_df["y_true"].astype(int)
        self.num_classes = self.labels.iloc[0].size

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        stay_id = self.stay_ids[idx]

        ehr_path = os.path.join(self.ehr_dir, f"{stay_id}")
        ehr_df = (
            pd.read_csv(ehr_path, index_col=False, encoding="utf-8")
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
        )

        # Convert to tensors
        ehr_tensor = torch.tensor(ehr_df.values, dtype=torch.float32)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float)

        return ehr_tensor, label


# define and create dataset (using processed EHR from MIMIC III data)
class ReportDataset(Dataset[tuple[str, Tensor]]):
    def __init__(self, root_dir: str, drop_duplicates: bool = False):
        super().__init__()
        self.root_dir = root_dir
        self.drop_duplicates = drop_duplicates
        self.report_dir = os.path.join(root_dir, "Report/")
        label_path = os.path.join(root_dir, "labels.csv")

        self.labels_df = pd.read_csv(label_path, index_col=False)
        self.stay_ids = self.labels_df["stay"].astype(str)
        if "phenotyping" in root_dir:
            self.labels = self.labels_df.iloc[:, 2:].astype(int)
        else:
            self.labels = self.labels_df["y_true"].astype(int)
        self.num_classes = self.labels.iloc[0].size

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index: int) -> tuple[str, Tensor]:
        stay_id = self.stay_ids[index]

        if "mimic-iv" in self.root_dir:
            report_path = os.path.join(
                self.report_dir, os.path.splitext(str(stay_id))[0] + ".txt"
            )
            with open(report_path, "r", encoding="utf-8") as f:
                report = f.read()
                reports = [report] * 48
        else:
            report_path = os.path.join(self.report_dir, str(stay_id))
            report_df = pd.read_csv(
                report_path, index_col=False, encoding="utf-8"
            ).ffill()
            if "phenotyping" in self.root_dir:
                reports = (
                    report_df.values.tolist()
                    if not self.drop_duplicates
                    else report_df.drop_duplicates().values.tolist()
                )
            else:
                reports = (
                    report_df["Note"].to_list()
                    if not self.drop_duplicates
                    else report_df["Note"].drop_duplicates().to_list()
                )
            reports = self.__remove_spaces_from_report(reports)
        reports = " [SEP] ".join(reports)
        text = "[CLS] " + reports

        label = torch.tensor(self.labels.iloc[index], dtype=torch.float32)

        return text, label

    def __remove_spaces_from_report(self, reports: list[str]) -> list[str]:
        reports = list(map(lambda x: x[::2], reports))
        return reports


class EHRAndReportDataset(Dataset[tuple[Tensor, str, Tensor, str]]):
    def __init__(self, root_dir: str) -> None:
        super().__init__()
        self.ehr_dir = os.path.join(root_dir, "EHR")
        self.report_dir = os.path.join(root_dir, "Report")

        label_path = os.path.join(root_dir, "labels.csv")
        self.labels_df = pd.read_csv(label_path, index_col=False)
        self.stay_ids = self.labels_df["stay"].astype(str)
        self.labels = self.labels_df["y_true"].astype(int)
        self.num_classes = self.labels.iloc[0].size

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, index) -> tuple[Tensor, str, Tensor, str]:
        """Get sample at index.

        Args:
            index: Index of data sample.

        Returns:
            tuple[Tensor, str, Tensor, str]: Tuple of EHR tensor, report text in str format, label tensor, and stay ID.
        """
        stay_id: str = self.stay_ids[index]  # pyright: ignore

        # Get report.
        if "mimic-iv" in self.report_dir:
            report_path = os.path.join(
                self.report_dir, os.path.splitext(str(stay_id))[0] + ".txt"
            )
            with open(report_path, "r", encoding="utf-8") as f:
                report = f.read()
                reports = [report]
        else:
            report_path = os.path.join(self.report_dir, str(stay_id))
            report_df = pd.read_csv(
                report_path, index_col=False, encoding="utf-8"
            ).ffill()
            if "phenotyping" in self.report_dir:
                reports = report_df.drop_duplicates().values.tolist()
            else:
                reports = report_df["Note"].drop_duplicates().to_list()
            reports = self.__remove_spaces_from_report(reports)

        reports = " [SEP] ".join(reports)
        text = "[CLS] " + reports

        # Get EHR data.
        ehr_path = os.path.join(self.ehr_dir, stay_id)
        ehr_df = (
            pd.read_csv(ehr_path, index_col=False, encoding="utf-8")
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
        )
        ehr_tensor = torch.tensor(ehr_df.values, dtype=torch.float32)
        label = torch.tensor(self.labels.iloc[index], dtype=torch.float32)

        return ehr_tensor, text, label, stay_id

    def __remove_spaces_from_report(self, reports: list[str]) -> list[str]:
        reports = list(map(lambda x: x[::2], reports))
        return reports


class EHRAndBioclinicalBERTEmbeddingsDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, root_dir: str, keep_label_as_long_tensor: bool = False) -> None:
        super().__init__()
        self.ehr_dir = os.path.join(root_dir, "EHR")
        self.emb_dir = os.path.join(
            root_dir, "Embeddings" if "phenotyping" not in root_dir else "Report"
        )

        label_path = os.path.join(root_dir, "labels.csv")
        self.labels_df = pd.read_csv(label_path, index_col=False)
        self.stay_ids = self.labels_df["stay"].astype(str)
        if "phenotyping" in root_dir:
            self.labels = self.labels_df.iloc[:, 2:].astype(int)
        else:
            self.labels = self.labels_df["y_true"].astype(int)
        self.num_classes = self.labels.iloc[0].size
        self.keep_label_as_long_tensor = keep_label_as_long_tensor

    def __len__(self) -> int:
        return len(self.labels_df)

    @override
    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor]:
        stay_id: str = self.stay_ids[index]  # pyright: ignore

        # Get report embeddings.
        emb_path = os.path.join(self.emb_dir, stay_id)
        emb_df = (
            pd.read_csv(emb_path, index_col=False, encoding="utf-8")
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
        )
        emb_tensor = torch.tensor(emb_df.values, dtype=torch.float32)

        # Get EHR data.
        ehr_path = os.path.join(self.ehr_dir, stay_id)
        ehr_df = (
            pd.read_csv(ehr_path, index_col=False, encoding="utf-8")
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
        )
        ehr_tensor = torch.tensor(ehr_df.values, dtype=torch.float32)

        if self.keep_label_as_long_tensor:
            label = torch.tensor(self.labels.iloc[index], dtype=torch.long).squeeze()
        else:
            label = torch.tensor(self.labels.iloc[index], dtype=torch.float32).squeeze()

        return ehr_tensor, emb_tensor, label


class BioclinicalBERTEmbeddingsDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, root_dir: str) -> None:
        super().__init__()
        self.emb_dir = os.path.join(root_dir, "Report")
        label_path = os.path.join(root_dir, "labels.csv")
        self.labels_df = pd.read_csv(label_path, index_col=False)
        self.stay_ids = self.labels_df["stay"].astype(str)
        if "phenotyping" in root_dir:
            self.labels = self.labels_df.iloc[:, 2:].astype(int)
        else:
            self.labels = self.labels_df["y_true"].astype(int)
        self.num_classes = self.labels.iloc[0].size

    def __len__(self) -> int:
        return len(self.labels_df)

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        stay_id: str = self.stay_ids[index]  # pyright: ignore

        # Get report embeddings.
        emb_path = os.path.join(self.emb_dir, stay_id)
        emb_df = (
            pd.read_csv(emb_path, index_col=False, encoding="utf-8")
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
        )
        emb_tensor = torch.tensor(emb_df.values, dtype=torch.float32)

        label = torch.tensor(
            self.labels.iloc[index].values, dtype=torch.float32
        ).squeeze()

        return emb_tensor, label


class EVIDEHRAndBioclinalBERTEmbeddingsDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, root_dir: str, keep_label_as_long_tensor: bool = False) -> None:
        super().__init__()
        self.ehr_dir = os.path.join(root_dir, "EHR")
        self.emb_dir = os.path.join(
            root_dir, "Embeddings" if "phenotyping" not in root_dir else "Report"
        )

        label_path = os.path.join(root_dir, "labels.csv")
        self.labels_df = pd.read_csv(label_path, index_col=False)
        self.stay_ids = self.labels_df["stay"].astype(str)
        if "phenotyping" in root_dir:
            self.labels = self.labels_df.iloc[:, 2:].astype(int)
        else:
            self.labels = self.labels_df["y_true"].astype(int)
        self.num_classes = self.labels.iloc[0].size
        self.keep_label_as_long_tensor = keep_label_as_long_tensor

    def __len__(self) -> int:
        return len(self.labels_df)

    @override
    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor]:
        stay_id: str = self.stay_ids[index]  # pyright: ignore

        # Get report embeddings.
        emb_path = os.path.join(self.emb_dir, stay_id)
        emb_df = (
            pd.read_csv(emb_path, index_col=False, encoding="utf-8")
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
        )
        ## Take only the first value.
        emb_tensor = torch.tensor(emb_df.values, dtype=torch.float32)[0, ...]

        # Get EHR data.
        ehr_path = os.path.join(self.ehr_dir, stay_id)
        ehr_df = (
            pd.read_csv(ehr_path, index_col=False, encoding="utf-8")
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
        )
        ## Take only the first value recorded.
        ehr_tensor = torch.tensor(ehr_df.values, dtype=torch.float32)[0, ...]

        if self.keep_label_as_long_tensor:
            label = torch.tensor(self.labels.iloc[index], dtype=torch.long).squeeze()
        else:
            label = torch.tensor(self.labels.iloc[index], dtype=torch.float32).squeeze()

        return ehr_tensor, emb_tensor, label

    @property
    def class_weights(self):
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(self.labels), y=self.labels
        ).tolist()
        return class_weights


# define 1 linear layer class
class Model_linear_1_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model_linear_1_layer, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())

    def forward(self, data):
        return self.model(data)


# define RNN class
class CustomRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super(CustomRNN, self).__init__()
        self.tanh = nn.Tanh()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.weight_ax = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(hidden_size, input_size if i == 0 else hidden_size)
                )
                for i in range(num_layers)
            ]
        )
        self.weight_aa = nn.ParameterList(
            [
                nn.Parameter(torch.randn(hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        self.bias_a = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)]
        )

        self.weight_hidden_bw_layers = nn.ParameterList(
            [
                nn.Parameter(torch.randn(hidden_size, hidden_size))
                for _ in range(num_layers - 1)
            ]
        )
        self.bias_hidden_bw_layers = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers - 1)]
        )

        self.weight_ya = nn.Parameter(torch.randn(input_size, hidden_size))
        self.bias_y = nn.Parameter(torch.zeros(input_size))

    @override
    def forward(self, data: Tensor) -> Tensor:
        for layer_num in range(self.num_layers):
            # Set initial hidden state to zero
            hidden_state = torch.zeros(
                data.shape[0], self.hidden_size, device=data.device
            )  # [B, H]
            outputs = []
            for time_step in range(data.shape[1]):
                # Extract 76 features of each time step
                input = data[:, time_step, :]  # [B, F]
                # Update hidden state
                hidden_state = self.tanh(
                    input @ self.weight_ax[layer_num].T
                    + hidden_state @ self.weight_aa[layer_num].T
                    + self.bias_a[layer_num]
                )  # [B, H]
                if layer_num < self.num_layers - 1:
                    output = (
                        hidden_state @ self.weight_hidden_bw_layers[layer_num].T
                        + self.bias_hidden_bw_layers[layer_num]
                    )  # [B, F]
                else:
                    output = hidden_state @ self.weight_ya.T + self.bias_y

                outputs.append(output)
            data = torch.stack(outputs, dim=1)  # [B, T, F]
        result = data[:, -1, :]
        return result

    @torch.no_grad()
    def predict(self, data: Tensor) -> Tensor:
        result = self.forward(data)
        return result.sigmoid()


# define LSTM class
class CustomLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super(CustomLSTM, self).__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.weight_ix = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(hidden_size, input_size if i == 0 else hidden_size)
                )
                for i in range(num_layers)
            ]
        )
        self.weight_ia = nn.ParameterList(
            [
                nn.Parameter(torch.randn(hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        self.bias_i = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)]
        )

        self.weight_fx = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(hidden_size, input_size if i == 0 else hidden_size)
                )
                for i in range(num_layers)
            ]
        )
        self.weight_fa = nn.ParameterList(
            [
                nn.Parameter(torch.randn(hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        self.bias_f = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)]
        )

        self.weight_gx = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(hidden_size, input_size if i == 0 else hidden_size)
                )
                for i in range(num_layers)
            ]
        )
        self.weight_ga = nn.ParameterList(
            [
                nn.Parameter(torch.randn(hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        self.bias_g = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)]
        )

        self.weight_ox = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(hidden_size, input_size if i == 0 else hidden_size)
                )
                for i in range(num_layers)
            ]
        )
        self.weight_oa = nn.ParameterList(
            [
                nn.Parameter(torch.randn(hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        self.bias_o = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers)]
        )

        self.weight_hidden_bw_layers = nn.ParameterList(
            [
                nn.Parameter(torch.randn(hidden_size, hidden_size))
                for _ in range(num_layers - 1)
            ]
        )
        self.bias_hidden_bw_layers = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size)) for _ in range(num_layers - 1)]
        )

        self.weight_ya = nn.Parameter(torch.randn(input_size, hidden_size))
        self.bias_y = nn.Parameter(torch.zeros(input_size))

    @override
    def forward(self, data: Tensor) -> Tensor:
        for layer_num in range(self.num_layers):
            # Set initial hidden state and cell state to zero
            hidden_state = torch.zeros(
                data.shape[0], self.hidden_size, device=data.device
            )  # [B, H]
            cell_state = torch.zeros(
                data.shape[0], self.hidden_size, device=data.device
            )  # [B, H]

            outputs = []
            for time_steps in range(data.shape[1]):
                # Extract 76 features of each time step
                input = data[:, time_steps, :]  # [B, F]
                # Calculate 4 gates
                gate_i_t = self.sigmoid(
                    input @ self.weight_ix[layer_num].T
                    + hidden_state @ self.weight_ia[layer_num].T
                    + self.bias_i[layer_num]
                )
                gate_f_t = self.sigmoid(
                    input @ self.weight_fx[layer_num].T
                    + hidden_state @ self.weight_fa[layer_num].T
                    + self.bias_f[layer_num]
                )
                gate_g_t = self.tanh(
                    input @ self.weight_gx[layer_num].T
                    + hidden_state @ self.weight_ga[layer_num].T
                    + self.bias_g[layer_num]
                )
                gate_o_t = self.sigmoid(
                    input @ self.weight_ox[layer_num].T
                    + hidden_state @ self.weight_oa[layer_num].T
                    + self.bias_o[layer_num]
                )

                # Update cell and hidden state
                cell_state = gate_f_t * cell_state + gate_i_t * gate_g_t  # [B, H]
                hidden_state = gate_o_t * self.tanh(cell_state)  # [B, H]

                if layer_num < self.num_layers - 1:
                    output = (
                        hidden_state @ self.weight_hidden_bw_layers[layer_num].T
                        + self.bias_hidden_bw_layers[layer_num]
                    )  # [B, F]
                else:
                    output = hidden_state @ self.weight_ya.T + self.bias_y
                outputs.append(output)
            data = torch.stack(outputs, dim=1)  # [B, T, F]
        result = data[:, -1, :]
        return result

    @torch.no_grad()
    def predict(self, data: Tensor) -> Tensor:
        result = self.forward(data)
        return result.sigmoid()


# Transformer block
class CustomTransformerBlock(nn.Module):
    def __init__(
        self,
        Trans_input_dim: int,
        Trans_n_heads: int,
        Trans_ff_dim: int,
        Trans_dropout: float,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            Trans_input_dim, Trans_n_heads, dropout=Trans_dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(Trans_input_dim, Trans_ff_dim),
            nn.ReLU(),
            nn.Linear(Trans_ff_dim, Trans_input_dim),
        )

        self.norm_1 = nn.LayerNorm(Trans_input_dim)
        self.norm_2 = nn.LayerNorm(Trans_input_dim)

        self.dropout = nn.Dropout(Trans_dropout)

    @override
    def forward(self, data: Tensor) -> Tensor:
        output_sa = self.self_attn(data, data, data)[0]
        output_sa_norm = self.norm_1(data + self.dropout(output_sa))

        output_ff = self.ffn(output_sa_norm)
        result = self.norm_2(data + self.dropout(output_ff))

        return result


# class Transformer
class CustomTransformer(nn.Module):
    def __init__(
        self,
        Trans_input_dim: int,
        Trans_n_heads: int,
        Trans_ff_dim: int,
        Trans_num_layers: int,
        Trans_max_len: int,
        Trans_dropout: float,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, Trans_input_dim))

        self.pos_embed = nn.Parameter(
            torch.zeros(1, Trans_max_len + 1, Trans_input_dim)
        )  # +1 for CLS token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(Trans_dropout)
        self.blocks = nn.ModuleList(
            [
                CustomTransformerBlock(
                    Trans_input_dim, Trans_n_heads, Trans_ff_dim, Trans_dropout
                )
                for _ in range(Trans_num_layers)
            ]
        )

    @override
    def forward(self, data: Tensor) -> Tensor:
        batch_size, seq_len, _ = data.shape

        # Add CLS tokens
        cls_token = self.cls_token.expand(batch_size, 1, -1)
        data = torch.cat([cls_token, data], dim=1)  # [batch, seq_len+1, d_model]

        # Add learnable positional embeddings
        pos_embed = self.pos_embed[:, : seq_len + 1, :]
        result = self.dropout(data + pos_embed)

        for block in self.blocks:
            result = block(result)

        return result[:, 0, :]  # CLS tokens

    @torch.no_grad()
    def predict(self, data: Tensor) -> Tensor:
        result = self.forward(data)
        return result.sigmoid()


class BERT(nn.Module):
    def __init__(
        self,
        model_name: Literal[
            "google-bert/bert-base-uncased",
            "dmis-lab/biobert-v1.1",
            "emilyalsentzer/Bio_ClinicalBERT",
        ],
        dropout_p: float = 0.5,
        n_fc_1: int = 256,
        n_fc_2: int = 48,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: BertModel = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout_p)

        # Batch-norm
        self.batchnorm_1 = nn.BatchNorm1d(n_fc_1, momentum=0.1)
        self.batchnorm_2 = nn.BatchNorm1d(n_fc_2, momentum=0.1)

        # Define the FC layers
        self.linear_1 = nn.Linear(self.model.config.hidden_size, n_fc_1)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(n_fc_1, n_fc_2)
        self.classifier = nn.Linear(n_fc_2, num_classes)

    def forward(self, x_text: list[str]) -> Tensor:
        inputs = self.tokenizer(
            x_text, return_tensors="pt", max_length=512, truncation=True, padding=True
        ).to("cuda")
        outputs = self.model(**inputs)
        bert_output = outputs[0][:, 0, :]

        # Pass through 1st FC layer.
        out_1 = self.relu(self.linear_1(bert_output))
        out_1 = self.batchnorm_1(out_1)
        out_1 = self.dropout(out_1)

        # Pass through 2nd FC layer.
        out_2 = self.relu(self.linear_2(out_1))
        out_2 = self.batchnorm_2(out_2)
        out_2 = self.dropout(out_2)

        # Pass through classifier output layer.
        out = self.classifier(out_2)

        return out

    @torch.no_grad()
    def predict(self, x_text: list[str]) -> Tensor:
        result = self.forward(x_text)
        return result.sigmoid()


class BERT_EHR(nn.Module):
    def __init__(
        self,
        static_size: int = 76,
        dropout_p: float = 0.5,
        n_fc_1: int = 256,
        n_fc_2: int = 48,
        temporal_conv: bool = False,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        self.model: BertModel = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        # Reduce BERT's output dimensionality.
        self.bert_output_reducer = nn.Linear(self.model.config.hidden_size, 50)

        # Static data dimensions
        self.static_size = static_size
        self.temporal_conv = (
            # nn.Conv1d(static_size, static_size, kernel_size=48, stride=1, padding=0)
            nn.Sequential(
                nn.Conv1d(static_size, static_size, kernel_size=3, stride=3),
                nn.ReLU(inplace=True),
                nn.Conv1d(static_size, static_size, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv1d(static_size, static_size, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv1d(static_size, static_size, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv1d(static_size, static_size, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
            )
            if temporal_conv
            else None
        )
        self.dropout = nn.Dropout(p=dropout_p)

        # Batch-norm
        self.batchnorm_1 = nn.BatchNorm1d(n_fc_1, momentum=0.1)
        self.batchnorm_2 = nn.BatchNorm1d(n_fc_2, momentum=0.1)

        # Define the FC layers
        self.linear_1 = nn.Linear(50 + static_size, n_fc_1)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(n_fc_1, n_fc_2)
        self.classifier = nn.Linear(n_fc_2, num_classes)

    def forward(self, x_text: list[str], x_static: Tensor) -> Tensor:
        inputs = self.tokenizer(
            x_text, return_tensors="pt", max_length=512, truncation=True, padding=True
        ).to("cuda")
        outputs = self.model(**inputs)

        # Extract the [CLS] token's representation (used for classification tasks)
        bert_output = outputs[0][:, 0, :]
        bert_output = self.bert_output_reducer(bert_output)

        # If we use a temporal conv net, perform temp conv. Otherwise take the mean
        # over the temporal axis of the static features.
        x_static = (
            self.temporal_conv(x_static.permute(0, 2, 1)).squeeze(dim=2)
            if isinstance(self.temporal_conv, nn.Sequential)
            else x_static.mean(dim=1, keepdim=False)
        )
        # Concatenate BERT's output and the static features.
        inputs = torch.cat([bert_output, x_static], dim=1)

        # Pass through 1st FC layer.
        out_1 = self.relu(self.linear_1(inputs))
        out_1 = self.batchnorm_1(out_1)
        out_1 = self.dropout(out_1)

        # Pass through 2nd FC layer.
        out_2 = self.relu(self.linear_2(out_1))
        out_2 = self.batchnorm_2(out_2)
        out_2 = self.dropout(out_2)

        # Pass through classifier output layer.
        out = self.classifier(out_2)

        return out

    @torch.no_grad()
    def predict(self, x_test: list[str], x_static: Tensor) -> Tensor:
        result = self.forward(x_test, x_static)
        return result.sigmoid()


class BERTEmbeddingsOnly(nn.Module):
    def __init__(
        self,
        emb_dim: int = 512,
        dropout_p: float = 0.5,
        n_fc_1: int = 256,
        n_fc_2: int = 48,
        num_classes: int = 1,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)

        # Batch-norm
        self.batchnorm_1 = nn.BatchNorm1d(n_fc_1, momentum=0.1)
        self.batchnorm_2 = nn.BatchNorm1d(n_fc_2, momentum=0.1)

        # Define the FC layers
        self.linear_1 = nn.Linear(emb_dim, n_fc_1)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(n_fc_1, n_fc_2)
        self.classifier = nn.Linear(n_fc_2, num_classes)

    def forward(self, x_emb: Tensor) -> Tensor:
        emb_sqz = x_emb.mean(dim=1, keepdim=False)

        # Pass through 1st FC layer.
        out_1 = self.relu(self.linear_1(emb_sqz))
        out_1 = self.batchnorm_1(out_1)
        out_1 = self.dropout(out_1)

        # Pass through 2nd FC layer.
        out_2 = self.relu(self.linear_2(out_1))
        out_2 = self.batchnorm_2(out_2)
        out_2 = self.dropout(out_2)

        # Pass through classifier output layer.
        out = self.classifier(out_2)

        return out

    @torch.no_grad()
    def predict(self, x_emb: Tensor) -> Tensor:
        result = self.forward(x_emb)
        return result.sigmoid()


class BERTEmbeddings_EHR(nn.Module):
    def __init__(
        self,
        static_size: int = 76,
        emb_dim: int = 512,
        dropout_p: float = 0.5,
        n_fc_1: int = 256,
        n_fc_2: int = 48,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)

        # Batch-norm
        self.batchnorm_1 = nn.BatchNorm1d(n_fc_1, momentum=0.1)
        self.batchnorm_2 = nn.BatchNorm1d(n_fc_2, momentum=0.1)

        # Define the FC layers
        self.linear_1 = nn.Linear(emb_dim + static_size, n_fc_1)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(n_fc_1, n_fc_2)
        self.classifier = nn.Linear(n_fc_2, num_classes)

    def forward(self, x_ehr: Tensor, x_emb: Tensor) -> Tensor:
        x_ehr = x_ehr.mean(dim=1, keepdim=False)
        x_emb = x_emb.mean(dim=1, keepdim=False)

        inputs = torch.cat([x_emb, x_ehr], dim=1)

        # Pass through 1st FC layer.
        out_1 = self.relu(self.linear_1(inputs))
        out_1 = self.batchnorm_1(out_1)
        out_1 = self.dropout(out_1)

        # Pass through 2nd FC layer.
        out_2 = self.relu(self.linear_2(out_1))
        out_2 = self.batchnorm_2(out_2)
        out_2 = self.dropout(out_2)

        # Pass through classifier output layer.
        out = self.classifier(out_2)

        return out

    @torch.no_grad()
    def predict(self, x_ehr: Tensor, x_emb: Tensor) -> Tensor:
        result = self.forward(x_ehr, x_emb)
        return result.sigmoid()


class BERTEmbeddings_EHR_TCN(nn.Module):
    def __init__(
        self,
        static_size: int = 76,
        emb_dim: int = 512,
        dropout_p: float = 0.5,
        n_fc_1: int = 256,
        n_fc_2: int = 48,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.ehr_tcn = nn.Sequential(
            nn.Conv1d(static_size, static_size, kernel_size=3, stride=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(static_size, static_size, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(static_size, static_size, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(static_size, static_size, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(static_size, static_size, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.emb_tcn = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, kernel_size=3, stride=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(emb_dim, emb_dim, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(emb_dim, emb_dim, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(emb_dim, emb_dim, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(emb_dim, emb_dim, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)

        # Batch-norm
        self.batchnorm_1 = nn.BatchNorm1d(n_fc_1, momentum=0.1)
        self.batchnorm_2 = nn.BatchNorm1d(n_fc_2, momentum=0.1)

        # Define the FC layers
        self.linear_1 = nn.Linear(emb_dim + static_size, n_fc_1)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(n_fc_1, n_fc_2)
        self.classifier = nn.Linear(n_fc_2, num_classes)

    def forward(self, x_ehr: Tensor, x_emb: Tensor) -> Tensor:
        ehr_sqz = self.ehr_tcn(x_ehr.permute(0, 2, 1)).squeeze(dim=2)
        emb_sqz = self.emb_tcn(x_emb.permute(0, 2, 1)).squeeze(dim=2)

        # Concatenate temporally convolved EHR and BioclinicalBERT embeddings
        inputs = torch.cat([ehr_sqz, emb_sqz], dim=1)

        # Pass through 1st FC layer.
        out_1 = self.relu(self.linear_1(inputs))
        out_1 = self.batchnorm_1(out_1)
        out_1 = self.dropout(out_1)

        # Pass through 2nd FC layer.
        out_2 = self.relu(self.linear_2(out_1))
        out_2 = self.batchnorm_2(out_2)
        out_2 = self.dropout(out_2)

        # Pass through classifier output layer.
        out = self.classifier(out_2)

        return out

    @torch.no_grad()
    def predict(self, x_ehr: Tensor, x_emb: Tensor) -> Tensor:
        result = self.forward(x_ehr, x_emb)
        return result.sigmoid()


# define 3 linear layers model
class Model_head_3_layers(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_l1_dim: int,
        output_l2_dim: int,
        output_l3_dim: int,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_l1_dim),
            nn.ReLU(),
            nn.Linear(output_l1_dim, output_l2_dim),
            nn.ReLU(),
            nn.Linear(output_l2_dim, output_l3_dim),
            nn.Sigmoid(),
        )

    def forward(self, data: Tensor) -> Tensor:
        data = data.squeeze(1)
        return self.model(data)

    @torch.no_grad()
    def predict(self, data: Tensor) -> Tensor:
        result = self.forward(data)
        return result.sigmoid()


# define 1 linear layer class
class Model_head_1_layer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())

    def forward(self, data: Tensor) -> Tensor:
        data = data.squeeze(1)
        return self.model(data)

    @torch.no_grad()
    def predict(self, data: Tensor) -> Tensor:
        result = self.forward(data)
        return result.sigmoid()


# DEBUG: Check whether the model works
if __name__ == "__main__":
    # PyTorch
    from torch.utils.data import DataLoader

    dataset = EHRAndReportDataset("./data/train_with_raw_report/")
    dataloader = DataLoader(dataset, batch_size=2)

    model = BERT_EHR().cuda()

    for ehr, report, label, stay_id in dataloader:
        ehr, label = ehr.cuda(), label.cuda()
        print(ehr, label)
        print(report, stay_id)
        out = model(report, ehr)
        print(out)
        break
