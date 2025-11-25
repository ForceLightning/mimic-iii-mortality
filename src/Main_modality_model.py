# Standard Library
import copy
import gc
import os
import random
from typing import Literal

# Third-Party
from jsonargparse import auto_cli
from rich.console import Console
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# Scientific Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

# PyTorch
import torch
from torch import GradScaler, Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

# First party imports
from Model_n_Dataset import (
    BERT,
    BERT_EHR,
    BERTEmbeddings_EHR,
    BERTEmbeddings_EHR_TCN,
    BERTEmbeddingsOnly,
    BioclinicalBERTEmbeddingsDataset,
    CustomLSTM,
    CustomRNN,
    CustomTransformer,
    DatasetType,
    EHRAndBioclinicalBERTEmbeddingsDataset,
    EHRAndReportDataset,
    EHRDataset,
    Model_head_1_layer,
    Model_head_3_layers,
    Model_linear_1_layer,
    ModelType,
    ReportDataset,
)
from utils.roc import auc_charts

SEED_CUS = 3407
CONSOLE = Console()


def set_seed(seed_cus):
    random.seed(seed_cus)
    np.random.seed(seed_cus)
    torch.manual_seed(seed_cus)
    torch.cuda.manual_seed(seed_cus)
    torch.cuda.manual_seed_all(seed_cus)


def worker_init_fn(worker_id):
    seed = SEED_CUS  # Or any constant
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


g = torch.Generator()
g.manual_seed(SEED_CUS)

# set parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data/"
CHECKPOINTS_DIR = "./checkpoints/"


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    model_type: ModelType,
    data_type: DatasetType,
    optimizer: Optimizer,
    criterion: nn.modules.loss._Loss,
    scaler: GradScaler,
    progress: Progress,
    epoch_task: TaskID,
):
    epoch_loss = 0.0
    num_batches = len(train_loader)
    batch_task = progress.add_task(
        "[blue]Training Mini-batches...", total=num_batches, loss=float("nan")
    )
    model.train()
    for i, data_tuple in enumerate(train_loader):
        optimizer.zero_grad()

        ehr: Tensor
        report: Tensor | list[str]
        label: Tensor
        num_samples: int = data_tuple[-1].shape[0]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            match (model_type, data_type):
                case (
                    ModelType.RNN
                    | ModelType.LSTM
                    | ModelType.TRANSFORMER
                    | ModelType.BERT,
                    DatasetType.EHR | DatasetType.REPORT,
                ):
                    data, label = data_tuple
                    if isinstance(data, Tensor):
                        data = data.to(DEVICE)
                    label = label.to(DEVICE).unsqueeze(1)

                    output = model(data)

                case (ModelType.BERT_EHR, DatasetType.EHR_AND_REPORT):
                    ehr, report, label, _stay_id = data_tuple
                    ehr = ehr.to(DEVICE)
                    if isinstance(report, Tensor):
                        report = report.to(DEVICE)
                    label = label.to(DEVICE).unsqueeze(1)

                    output = model(report, ehr)

                case ModelType.BERT_EMB_EHR_TCN, DatasetType.BCB_EMB_EHR:
                    ehr, emb, label = data_tuple
                    ehr = ehr.to(DEVICE)
                    emb = emb.to(DEVICE)
                    label = label.to(DEVICE).unsqueeze(1)

                    output = model(ehr, emb)

                case ModelType.BERT_EMB, DatasetType.BCB_EMB:
                    emb, label = data_tuple
                    emb = emb.to(DEVICE)
                    label = label.to(DEVICE).unsqueeze(1)

                    output = model(emb)

                case ModelType.BERT_EMB, DatasetType.PHENOTYPE_BCB_EMB:
                    data, label = data_tuple
                    if isinstance(data, Tensor):
                        data = data.to(DEVICE)
                    label = label.to(DEVICE)

                    output = model(data)

                case ModelType.BERT_EHR, DatasetType.PHENOTYPE_BCB_EMB_EHR:
                    ehr, emb, label = data_tuple
                    ehr = ehr.to(DEVICE)
                    emb = emb.to(DEVICE)
                    label = label.to(DEVICE)

                    output = model(ehr, emb)

                case _, _:
                    raise NotImplementedError(
                        f"Combination of {model_type}, {data_type} not implemented!"
                    )

            loss = criterion(output, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        current_loss = loss.detach().cpu().item()
        epoch_loss += current_loss

        progress.update(epoch_task, advance=num_samples)
        progress.update(batch_task, advance=1, loss=current_loss)

    progress.stop_task(batch_task)
    progress.remove_task(batch_task)
    return epoch_loss


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    model_type: ModelType,
    data_type: DatasetType,
    criterion: nn.modules.loss._Loss,
    progress: Progress,
    epoch_task: TaskID,
):
    output_test = []
    label_test = []

    model.eval()

    num_batches = len(test_loader)
    batch_task = progress.add_task(
        "[blue]Validation Mini-batches...", total=num_batches, loss=float("nan")
    )
    epoch_loss = 0.0
    progress.console.print("start testing")
    for i, data_tuple in enumerate(test_loader):
        num_samples: int = data_tuple[-1].shape[0]
        match (model_type, data_type):
            case (
                ModelType.RNN | ModelType.LSTM | ModelType.TRANSFORMER | ModelType.BERT,
                DatasetType.EHR | DatasetType.REPORT,
            ):
                data, label = data_tuple
                if isinstance(data, Tensor):
                    data = data.to(DEVICE)
                label = label.to(DEVICE).unsqueeze(1)

                output = model(data)

            case (ModelType.BERT_EHR, DatasetType.EHR_AND_REPORT):
                ehr, report, label, _stay_id = data_tuple
                ehr = ehr.to(DEVICE)
                if isinstance(report, Tensor):
                    report = report.to(DEVICE)
                label = label.to(DEVICE).unsqueeze(1)

                output = model(report, ehr)

            case ModelType.BERT_EMB_EHR_TCN, DatasetType.BCB_EMB_EHR:
                ehr, emb, label = data_tuple
                ehr = ehr.to(DEVICE)
                emb = emb.to(DEVICE)
                label = label.to(DEVICE).unsqueeze(1)

                output = model(ehr, emb)

            case ModelType.BERT_EMB, DatasetType.BCB_EMB:
                emb, label = data_tuple
                emb = emb.to(DEVICE)
                label = label.to(DEVICE).unsqueeze(1)

                output = model(emb)

            case ModelType.BERT_EMB, DatasetType.PHENOTYPE_BCB_EMB:
                data, label = data_tuple
                if isinstance(data, Tensor):
                    data = data.to(DEVICE)
                label = label.to(DEVICE)

                output = model(data)

            case ModelType.BERT_EHR, DatasetType.PHENOTYPE_BCB_EMB_EHR:
                ehr, emb, label = data_tuple
                ehr = ehr.to(DEVICE)
                emb = emb.to(DEVICE)
                label = label.to(DEVICE)

                output = model(ehr, emb)

            case _, _:
                raise NotImplementedError(
                    f"Combination of {model_type}, {data_type} not implemented!"
                )

        loss = criterion(output, label)
        current_loss = loss.detach().cpu().item()
        epoch_loss += current_loss

        output_test.append(output.detach().cpu())
        label_test.append(label.detach().cpu())
        progress.update(batch_task, advance=1, loss=current_loss)
        progress.update(epoch_task, advance=num_samples)

    progress.stop_task(batch_task)
    progress.remove_task(batch_task)
    output_test = torch.cat(output_test)
    label_test = torch.cat(label_test)

    return output_test, label_test


@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    model_type: ModelType,
    data_type: DatasetType,
    model_path: str,
    model_name_list: list[str],
):
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
    ) as progress:
        output_test = []
        label_test = []

        ### load best model
        model.load_state_dict(
            torch.load(
                os.path.join(model_path, f"{"_".join(model_name_list)}.ckpt"),
                map_location=DEVICE,
            )
        )
        model.eval()

        ### testing
        task = progress.add_task("Test batches", total=len(test_loader))
        for data_tuple in test_loader:
            match (model_type, data_type):
                case (
                    ModelType.RNN
                    | ModelType.LSTM
                    | ModelType.TRANSFORMER
                    | ModelType.BERT,
                    DatasetType.EHR | DatasetType.REPORT,
                ):
                    data, label = data_tuple
                    if isinstance(data, Tensor):
                        data = data.to(DEVICE)
                    label = label.to(DEVICE).unsqueeze(1)

                    output = model(data)

                case (ModelType.BERT_EHR, DatasetType.EHR_AND_REPORT):
                    ehr, report, label, _stay_id = data_tuple
                    ehr = ehr.to(DEVICE)
                    if isinstance(report, Tensor):
                        report = report.to(DEVICE)
                    label = label.to(DEVICE).unsqueeze(1)

                    output = model(report, ehr)

                case ModelType.BERT_EMB, DatasetType.BCB_EMB:
                    emb, label = data_tuple
                    emb = emb.to(DEVICE)
                    label = label.to(DEVICE).unsqueeze(1)

                    output = model(emb)

                case ModelType.BERT_EMB_EHR_TCN, DatasetType.BCB_EMB_EHR:
                    ehr, emb, label = data_tuple
                    ehr = ehr.to(DEVICE)
                    emb = emb.to(DEVICE)
                    label = label.to(DEVICE).unsqueeze(1)

                    output = model(ehr, emb)

                case ModelType.BERT_EMB, DatasetType.PHENOTYPE_BCB_EMB:
                    data, label = data_tuple
                    if isinstance(data, Tensor):
                        data = data.to(DEVICE)
                    label = label.to(DEVICE)

                    output = model(data)

                case ModelType.BERT_EHR, DatasetType.PHENOTYPE_BCB_EMB_EHR:
                    ehr, emb, label = data_tuple
                    ehr = ehr.to(DEVICE)
                    emb = emb.to(DEVICE)
                    label = label.to(DEVICE)

                    output = model(ehr, emb)

                case _, _:
                    raise NotImplementedError(
                        f"Combination of {model_type}, {data_type} not implemented!"
                    )

            # append all outputs
            output_test.append(output.detach().cpu())

            # append all labels
            label_test.append(label.detach().cpu())
            progress.advance(task)

        # create list for all outputs
        output_test = torch.cat(output_test).detach().cpu()

        # create list for all label
        label_test = torch.cat(label_test).detach().cpu()

        # calculate AUROC and AUPRC
        roc_auc, ap_score, *_ = data_type.get_eval_fn()
        auroc = roc_auc(label_test, output_test)
        auprc = ap_score(label_test, output_test)

        progress.stop()

    return output_test, label_test, auroc, auprc


def train_and_validate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_type: ModelType,
    data_type: DatasetType,
    num_epochs: int,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    criterion: nn.modules.loss._Loss,
    result_path: str,
):
    epoch_all = []
    auroc_all = []
    best_auroc = -0.1
    best_model = {}
    scaler = GradScaler()  # Automatic Mixed Precision

    # training loop
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
        TextColumn("Loss: {task.fields[loss]:0.4f}"),
    ) as progress:
        batch_size = train_loader.batch_size if train_loader.batch_size else 16
        total_train_samples = (
            len(train_loader.dataset)  # pyright: ignore[reportArgumentType]
            // batch_size
            * num_epochs
            * batch_size
        )
        total_test_samples = (
            len(test_loader.dataset) * num_epochs  # pyright: ignore[reportArgumentType]
        )
        total_samples = total_train_samples + total_test_samples

        epoch_task = progress.add_task(
            "[green]Total samples", total=total_samples, loss=float("nan")
        )
        for epoch in range(num_epochs):
            # start training
            progress.console.print("start training")
            model.train()

            num_batches = len(train_loader)
            epoch_loss = train_epoch(
                model,
                train_loader,
                model_type,
                data_type,
                optimizer,
                criterion,
                scaler,
                progress,
                epoch_task,
            )

            avg_epoch_loss = epoch_loss / num_batches

            scheduler.step()

            output_test, label_test = validate_epoch(
                model,
                test_loader,
                model_type,
                data_type,
                criterion,
                progress,
                epoch_task,
            )

            # Calculate AUROC or AUPRC
            roc_auc, *_ = data_type.get_eval_fn()
            auroc = roc_auc(label_test, output_test)
            # auprc = average_precision_score(label_test, output_test)

            # saving results
            epoch_all.append(epoch)
            auroc_all.append(auroc)
            df_test_auroc = pd.DataFrame(
                {"Epoch": epoch_all, "Testing AUROC": auroc_all}
            )
            df_test_auroc.to_csv(
                os.path.join(result_path, "Testing_all_AUROC.csv"), index=False
            )
            # print results
            progress.console.print(f"Testing AUROC is {100 * auroc}%")
            if auroc > best_auroc:
                best_auroc = auroc
                best_model = copy.deepcopy(model.state_dict())
            progress.update(epoch_task, loss=avg_epoch_loss)

    return best_model


def find_optimal_thresholds(label_test: Tensor, output: Tensor) -> Tensor:
    n_classes = label_test.shape[1]
    thresholds = []

    for i in range(n_classes):
        fpr, tpr, thresh = roc_curve(label_test[:, i], output[:, i])
        youden_j = tpr - fpr
        best_thresh = thresh[np.argmax(youden_j)]
        thresholds.append(best_thresh)

    return torch.as_tensor(thresholds, device=output.device, dtype=output.dtype)


def main(
    model_type: ModelType,
    data_type: DatasetType,
    data_dir: str = DATA_DIR,
    checkpoints_dir: str = CHECKPOINTS_DIR,
    batch_size: int = 64,
    hidden_size: int = 32,
    num_layers: int = 3,
    num_epoch: int = 1,
    lr: float = 5e-3,
    trans_input_dim: int = 512,
    trans_n_heads: int = 8,
    trans_ff_dim: int = 2048,
    trans_dropout: float = 0.1,
    trans_num_layers: int = 12,
    trans_max_len: int = 512,
    output_l1_dim: int = 256,
    output_l2_dim: int = 128,
    output_l3_dim: int = 1,
    bert_use_temporal_conv: bool = False,
    bert_model_str: Literal[
        "google-bert/bert-base-uncased",
        "dmis-lab/biobert-v1.1",
        "emilyalsentzer/Bio_ClinicalBERT",
    ] = "google-bert/bert-base-uncased",
    seed_everything: int = SEED_CUS,
    show_auc_plots: bool = False,
):
    set_seed(seed_everything)
    # define train and test dataset

    Dataset = data_type.get_dataset_class()

    assert Dataset is not None

    train_dir, test_dir = data_type.get_train_test_dir()

    train_dataset = Dataset(os.path.join(data_dir, train_dir))
    test_dataset = Dataset(os.path.join(data_dir, test_dir))

    num_classes = train_dataset.num_classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=(
            16
            if data_type
            in [
                DatasetType.BCB_EMB,
                DatasetType.BCB_EMB_EHR,
                DatasetType.PHENOTYPE_BCB_EMB,
                DatasetType.PHENOTYPE_BCB_EMB_EHR,
            ]
            else 8
        ),
        persistent_workers=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=(
            16
            if data_type
            in [
                DatasetType.BCB_EMB,
                DatasetType.BCB_EMB_EHR,
                DatasetType.PHENOTYPE_BCB_EMB,
                DatasetType.PHENOTYPE_BCB_EMB_EHR,
            ]
            else 8
        ),
        persistent_workers=True,
    )

    ### model saving
    # change path accroding to the model structure (Encoder/Discriminator/Scheduler) and data (partial or full)
    model_type_detailed = str(model_type)
    if model_type == ModelType.BERT:
        model_type_detailed = bert_model_str.split("/")[1]
    model_path = os.path.join(
        f"{checkpoints_dir}/Dataset_{str(data_type)}_Model_{str(model_type)}{"_tcn" if bert_use_temporal_conv else ""}_epoch_{str(num_epoch)}_CosLR_lr_{str(lr)}_seed{str(seed_everything)}/",
        model_type_detailed,
    )
    if not os.path.exists(model_path):
        # Create a new directory because it does not exist
        os.makedirs(model_path)
        CONSOLE.print(
            f"The new model directory is created: {os.path.normpath(model_path)}"
        )

    ### result saving
    # change path accroding to the model structure (Encoder/Discriminator/Scheduler) and data (partial or full)
    result_path = os.path.join(
        f"{checkpoints_dir}/Dataset_{str(data_type)}_Model_{str(model_type)}{"_tcn" if bert_use_temporal_conv else ""}_epoch_{str(num_epoch)}_CosLR_lr_{str(lr)}_seed{str(seed_everything)}/",
        model_type_detailed,
    )
    if not os.path.exists(result_path):
        # Create a new directory because it does not exist
        os.makedirs(result_path)
        CONSOLE.print(
            f"The new result directory is created: {os.path.normpath(result_path)}"
        )

    # set criterion
    criterion = nn.BCEWithLogitsLoss()

    # define models
    model_sequential: nn.Module
    match (model_type, train_dataset):
        case ModelType.RNN, EHRDataset() | ReportDataset():
            model_sequential = nn.Sequential(
                CustomRNN(train_dataset[0][0].shape[-1], hidden_size, num_layers),
                Model_head_3_layers(
                    train_dataset[0][0].shape[-1],
                    output_l1_dim,
                    output_l2_dim,
                    output_l3_dim,
                ),
            )
            model_name_list = ["Model_RNN", "Model_head_linear"]

        case ModelType.LSTM, EHRDataset() | ReportDataset():
            model_sequential = nn.Sequential(
                CustomLSTM(train_dataset[0][0].shape[-1], hidden_size, num_layers),
                Model_head_3_layers(
                    train_dataset[0][0].shape[-1],
                    output_l1_dim,
                    output_l2_dim,
                    output_l3_dim,
                ),
            )
            model_name_list = ["Model_LSTM", "Model_head_linear"]

        case (
            ModelType.TRANSFORMER,
            EHRDataset() | ReportDataset(),
        ):
            model_sequential = nn.Sequential(
                Model_linear_1_layer(train_dataset[0][0].shape[-1], trans_input_dim),
                CustomTransformer(
                    trans_input_dim,
                    trans_n_heads,
                    trans_ff_dim,
                    trans_num_layers,
                    trans_max_len,
                    trans_dropout,
                ),
                Model_head_1_layer(trans_input_dim, output_l3_dim),
            )
            model_name_list = ["Model_linear", "Model_Transformer", "Model_head_linear"]

        case ModelType.BERT, ReportDataset():
            model_sequential = BERT(
                bert_model_str,
                trans_dropout,
                output_l1_dim,
                output_l2_dim,
                train_dataset[0][-1].numel(),
            )
            model_name_list = [model_type_detailed]

        case ModelType.BERT, BioclinicalBERTEmbeddingsDataset():
            model_sequential = BERT(
                bert_model_str,
                trans_dropout,
                output_l1_dim,
                output_l2_dim,
                train_dataset[0][-1].numel(),
            )
            model_name_list = [model_type_detailed]

        case ModelType.BERT_EHR, EHRAndReportDataset():
            print(model_type, train_dataset[0][0].shape)
            model_sequential = BERT_EHR(
                train_dataset[0][0].shape[-1],
                trans_dropout,
                output_l1_dim,
                output_l2_dim,
                bert_use_temporal_conv,
                train_dataset[0][2].numel(),
            )
            model_name_list = [model_type_detailed]

        case ModelType.BERT_EMB_EHR_TCN, EHRAndBioclinicalBERTEmbeddingsDataset():
            model_sequential = BERTEmbeddings_EHR_TCN(
                train_dataset[0][0].shape[-1],
                train_dataset[0][1].shape[-1],
                trans_dropout,
                output_l1_dim,
                output_l2_dim,
                train_dataset[0][-1].numel(),
            )
            model_name_list = [model_type_detailed]

        case ModelType.BERT_EMB, BioclinicalBERTEmbeddingsDataset():
            model_sequential = BERTEmbeddingsOnly(
                train_dataset[0][0].shape[-1],
                trans_dropout,
                output_l1_dim,
                output_l2_dim,
                train_dataset[0][-1].numel(),
            )
            model_name_list = [model_type_detailed]

        case ModelType.BERT_EHR, EHRAndBioclinicalBERTEmbeddingsDataset():
            model_sequential = BERTEmbeddings_EHR(
                train_dataset[0][0].shape[-1],
                train_dataset[0][1].shape[-1],
                trans_dropout,
                output_l1_dim,
                output_l2_dim,
                train_dataset[0][2].numel(),
            )
            model_name_list = [model_type_detailed]

        case _, _:
            raise NotImplementedError(
                f"Combination of {model_type}, {data_type} is not implemented!"
            )

    model_sequential.to(DEVICE)
    ckpt_path = os.path.join(model_path, f"{"_".join(model_name_list)}.ckpt")

    if not os.path.exists(ckpt_path):
        params = [
            param for param in model_sequential.parameters() if param.requires_grad
        ]
        optimizer = torch.optim.Adam(
            params, lr=lr, betas=(0.5, 0.99), weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epoch
        )

        best_model = train_and_validate(
            model_sequential,
            train_loader,
            test_loader,
            model_type,
            data_type,
            num_epoch,
            optimizer,
            scheduler,
            criterion,
            result_path,
        )

        ### loading best model weights
        model_sequential.load_state_dict(best_model)
        ### saving trained model
        CONSOLE.print("...saving model...")
        torch.save(model_sequential.state_dict(), ckpt_path)
    else:
        CONSOLE.print(f"Loading model weights from {os.path.normpath(ckpt_path)}")
        model_sequential.load_state_dict(
            torch.load(ckpt_path, weights_only=True, map_location=DEVICE)
        )

    # define testing process after training completion and model saving
    gc.collect()
    CONSOLE.print("start final testing")

    output_test, label_test, auroc, auprc = test(
        model_sequential,
        test_loader,
        model_type,
        data_type,
        model_path,
        model_name_list,
    )

    ### saving results
    df_test_acc = pd.DataFrame({"Testing AUROC": [auroc]})
    df_test_acc.to_csv(os.path.join(result_path, "Testing_best_AUROC.csv"), index=False)

    ### saving results
    df_test_acc = pd.DataFrame({"Testing AUPRC": [auprc]})
    df_test_acc.to_csv(os.path.join(result_path, "Testing_best_AUPRC.csv"), index=False)

    # To find the cut-off (threshold) value that gives the best classification performance from an AUROC
    best_threshold = find_optimal_thresholds(label_test, output_test)
    CONSOLE.print(f"Best threshold: {best_threshold}")

    # apply threshold on output to calculate following metrics
    # this one needs to be adjusted, regarding the thres value, based on the AUROC maybe?
    output_thresholded = (
        output_test >= best_threshold
    ).int()  # values > 0.1 → 1, else → 0

    table = Table(title=f"{model_type_detailed} metrics")
    table.add_column("AUROC", justify="right", style="cyan", no_wrap=True)
    table.add_column("AUPRC", justify="right", style="cyan", no_wrap=True)
    accuracy = 0

    # calculate accuracy
    if num_classes == 1:
        count = 0
        for i in range(label_test.size(dim=0)):
            if output_thresholded[i] == label_test[i]:
                count = count + 1

        accuracy = count / label_test.size(dim=0)

        ### print results
        # CONSOLE.print(
        #     f"Testing accuracy is {100 * (count / label_test.size(dim=0)):0.2f}%"
        # )
        table.add_column("Accuracy", justify="right", style="cyan", no_wrap=True)
        table.add_column("PPV", justify="right", style="cyan", no_wrap=True)
        table.add_column("F1", justify="right", style="cyan", no_wrap=True)

        ### saving results
        df_test_acc = pd.DataFrame({"Testing Acc": [count / label_test.size(dim=0)]})
        df_test_acc.to_csv(
            os.path.join(result_path, "Testing_best_Acc.csv"), index=False
        )
    else:
        table.add_column("PPV", justify="right", style="cyan", no_wrap=True)
        table.add_column("Recall", justify="right", style="cyan", no_wrap=True)
        table.add_column("F1", justify="right", style="cyan", no_wrap=True)

    _auroc_score, _ap_score, prec_score, rec_score, f1 = data_type.get_eval_fn()
    precision = prec_score(label_test, output_thresholded)
    recall = rec_score(label_test, output_thresholded)
    f1_score = f1(label_test, output_thresholded)

    if num_classes == 1:
        table.add_row(
            f"{auroc:.4f}",
            f"{auprc:.4f}",
            f"{accuracy:.4f}",
            f"{precision:.4f}",
            f"{f1_score:.4f}",
        )
    else:
        table.add_row(
            f"{auroc:.4f}",
            f"{auprc:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1_score:.4f}",
        )

    CONSOLE.print(table)

    ### saving results
    df_test_pre = pd.DataFrame({"Testing Pre": [precision]})
    df_test_pre.to_csv(os.path.join(result_path, "Testing_best_Pre.csv"), index=False)

    df_test_rec = pd.DataFrame({"Testing recall": [recall]})
    df_test_rec.to_csv(os.path.join(result_path, "Testing_best_Rec.csv"), index=False)

    df_test_f1 = pd.DataFrame({"Testing F1": [f1_score]})
    df_test_f1.to_csv(os.path.join(result_path, "Testing_best_F1.csv"), index=False)

    if show_auc_plots:
        model_str = (
            str(model_type) if not bert_use_temporal_conv else str(model_type) + "_TCN"
        )
        model_str = model_str + f"_seed{seed_everything}"

        fig = auc_charts(label_test, output_test, model_str)
        fig.savefig(
            f"./output/{model_str}.png",
        )
        plt.show(block=True)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
