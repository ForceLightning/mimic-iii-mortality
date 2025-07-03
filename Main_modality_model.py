# import functions
# Standard Library
import copy
import gc
import os

# define fixed seed
import random
from argparse import ArgumentParser
from typing import Literal

# Third-Party
from Model_n_Dataset import (
    BioClinicalBERT,
    CustomLSTM,
    CustomRNN,
    CustomTransformer,
    EHRAndReportDataset,
    EHRDataset,
    Model_head_1_layer,
    Model_head_3_layers,
    Model_linear_1_layer,
    ReportDataset,
)
from rich.progress import Progress

# Scientific Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# PyTorch
import torch
from torch import GradScaler, Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

SEED_CUS = 3407


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
    model_type: str,
    data_type: str,
    optimizer: Optimizer,
    criterion: nn.modules.loss._Loss,
    scaler: GradScaler,
    epoch: int,
    progress: Progress,
):
    epoch_loss = 0.0
    num_batches = len(train_loader)
    batch_task = progress.add_task("[blue]Training Mini-batches...", total=num_batches)
    model.train()
    for i, data_tuple in enumerate(train_loader):
        optimizer.zero_grad()

        ehr: Tensor
        report: Tensor
        label: Tensor

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            match (model_type, data_type):
                case "RNN" | "LSTM" | "Transformer", "EHR" | "Report":
                    data, label = data_tuple
                    data = data.to(DEVICE)
                    label = label.to(DEVICE).unsqueeze(1)

                    output = model(data)
                case "BioClinicalBERT", "EHRAndReport":
                    ehr, report, label, _stay_id = data_tuple
                    ehr = ehr.to(DEVICE)
                    label = label.to(DEVICE).unsqueeze(1)

                    output = model(report, ehr)
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

        progress.console.print(f"Epoch: {epoch + 1}, Batch: {i}, Loss: {current_loss}")
        progress.advance(batch_task)

    progress.stop_task(batch_task)
    progress.remove_task(batch_task)
    return epoch_loss


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    model_type: str,
    data_type: str,
    epoch: int,
    criterion: nn.modules.loss._Loss,
    progress: Progress,
):
    output_test = []
    label_test = []

    model.eval()

    num_batches = len(test_loader)
    batch_task = progress.add_task(
        "[blue]Validation Mini-batches...", total=num_batches
    )
    epoch_loss = 0.0
    progress.console.print("start testing")
    for i, data_tuple in enumerate(test_loader):
        match (model_type, data_type):
            case "RNN" | "LSTM" | "Transformer", "EHR" | "Report":
                data, label = data_tuple
                data = data.to(DEVICE)
                label = label.to(DEVICE).unsqueeze(1)

                output = model(data)
            case "BioClinicalBERT", "EHRAndReport":
                ehr, report, label, _stay_id = data_tuple
                ehr = ehr.to(DEVICE)
                label = label.to(DEVICE).unsqueeze(1)

                output = model(report, ehr)
            case _, _:
                raise NotImplementedError(
                    f"Combination of {model_type}, {data_type} not implemented!"
                )
        progress.advance(batch_task)

        loss = criterion(output, label)
        current_loss = loss.detach().cpu().item()
        epoch_loss += current_loss
        progress.console.print(f"Epoch: {epoch + 1}, Batch: {i}, Loss: {current_loss}")

        output_test.append(output.detach().cpu())
        label_test.append(label.detach().cpu())

    progress.stop_task(batch_task)
    progress.remove_task(batch_task)
    output_test = torch.cat(output_test)
    label_test = torch.cat(label_test)

    return output_test, label_test


@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    model_type: str,
    data_type: str,
    model_path: str,
    model_name_list: list[str],
):
    with Progress() as progress:
        output_test = []
        label_test = []

        ### load best model
        model.load_state_dict(
            torch.load(os.path.join(model_path, f"{"_".join(model_name_list)}.ckpt"))
        )
        model.to(DEVICE)
        model.eval()

        ### testing
        task = progress.add_task("Test batches", total=len(test_loader))
        for i, data_tuple in enumerate(test_loader):
            match (model_type, data_type):
                case "RNN" | "LSTM" | "Transformer", "EHR" | "Report":
                    data, label = data_tuple
                    data = data.to(DEVICE)
                    label = label.to(DEVICE).unsqueeze(1)

                    output = model(data)
                case "BioClinicalBERT", "EHRAndReport":
                    ehr, report, label, _stay_id = data_tuple
                    ehr = ehr.to(DEVICE)
                    label = label.to(DEVICE).unsqueeze(1)

                    output = model(report, ehr)
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
        auroc = roc_auc_score(label_test, output_test)
        auprc = average_precision_score(label_test, output_test)

        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}")

        progress.stop()

    return output_test, label_test, auroc, auprc


def train_and_validate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_type: str,
    data_type: str,
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
    with Progress(refresh_per_second=1, speed_estimate_period=30) as progress:
        epoch_task = progress.add_task("[green]Epochs", total=num_epochs)
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
                epoch,
                progress,
            )

            avg_epoch_loss = epoch_loss / num_batches
            progress.console.print(
                f"Number of epoch completed: {epoch + 1}, Epoch Loss: {avg_epoch_loss}"
            )

            scheduler.step()

            output_test, label_test = validate_epoch(
                model, test_loader, model_type, data_type, epoch, criterion, progress
            )

            # Calculate AUROC or AUPRC
            auroc = roc_auc_score(label_test, output_test)
            # auprc = average_precision_score(label_test, output_test)

            # saving results
            epoch_all.append(epoch)
            auroc_all.append(auroc)
            df_test_auroc = pd.DataFrame(
                {"Epoch": epoch_all, "Testing AUROC": auroc_all}
            )
            df_test_auroc.to_csv(result_path + "Testing_all_AUROC.csv", index=False)
            # print results
            progress.console.print(f"Testing AUROC is {100 * auroc}%")
            if auroc > best_auroc:
                best_auroc = auroc
                best_model = copy.deepcopy(model.state_dict())
            progress.advance(epoch_task)

    return best_model


def main(
    model_type: Literal["RNN", "LSTM", "Transformer", "BioClinicalBERT"],
    data_type: Literal["EHR", "Report", "EHRAndReport"],
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
):
    set_seed(SEED_CUS)
    # define train and test dataset
    if data_type == "EHR":
        Dataset = EHRDataset
    elif data_type == "Report":
        Dataset = ReportDataset
    elif data_type == "EHRAndReport":
        Dataset = EHRAndReportDataset

    assert Dataset is not None

    train_dataset = Dataset(os.path.join(data_dir, "train_with_raw_report"))
    test_dataset = Dataset(os.path.join(data_dir, "test_with_raw_report"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
    )

    ### model saving
    # change path accroding to the model structure (Encoder/Discriminator/Scheduler) and data (partial or full)
    model_path = f"{checkpoints_dir}/Dataset_{str(data_type)}_Model_{str(model_type)}{"_tcn" if bert_use_temporal_conv else ""}_epoch_{str(num_epoch)}_CosLR_lr_{str(lr)}_seed{str(SEED_CUS)}/"
    if not os.path.exists(model_path):
        # Create a new directory because it does not exist
        os.makedirs(model_path)
        print("The new model directory is created!")

    ### result saving
    # change path accroding to the model structure (Encoder/Discriminator/Scheduler) and data (partial or full)
    result_path = f"{checkpoints_dir}/Dataset_{str(data_type)}_Model_{str(model_type)}{"_tcn" if bert_use_temporal_conv else ""}_epoch_{str(num_epoch)}_CosLR_lr_{str(lr)}_seed{str(SEED_CUS)}/"
    if not os.path.exists(result_path):
        # Create a new directory because it does not exist
        os.makedirs(result_path)
        print("The new result directory is created!")

    # set criterion
    criterion = nn.BCEWithLogitsLoss()

    # define models
    model_sequential: nn.Module
    match (model_type, train_dataset):
        case "RNN", EHRDataset() | ReportDataset():
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
        case "LSTM", EHRDataset() | ReportDataset():
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
            "Transformer",
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
        case "BioClinicalBERT", EHRAndReportDataset():
            model_sequential = BioClinicalBERT(
                train_dataset[0][0].shape[-1],
                trans_dropout,
                output_l1_dim,
                output_l2_dim,
                bert_use_temporal_conv,
            )

            model_name_list = ["BioClinicalBERT"]
        case _, _:
            raise NotImplementedError(
                f"Combination of {model_type}, {data_type} is not implemented!"
            )

    model_sequential.to(DEVICE)

    params = [param for param in model_sequential.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

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
    print("...saving model...")
    torch.save(
        model_sequential.state_dict(),
        os.path.join(model_path, f"{"_".join(model_name_list)}.ckpt"),
    )

    # define testing process after training completion and model saving
    gc.collect()
    print("start final testing")

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
    df_test_acc.to_csv(result_path + "Testing_best_AUROC.csv", index=False)

    ### saving results
    df_test_acc = pd.DataFrame({"Testing AUPRC": [auprc]})
    df_test_acc.to_csv(result_path + "Testing_best_AUPRC.csv", index=False)

    # To find the cut-off (threshold) value that gives the best classification performance from an AUROC
    fpr, tpr, thresholds = roc_curve(label_test, output_test)
    # Youden's J statistic
    j_scores = tpr - fpr
    j_max_idx = j_scores.argmax()
    best_threshold = thresholds[j_max_idx]
    print(f"Best threshold: {best_threshold}")

    # apply threshold on output to calculate following metrics
    # this one needs to be adjusted, regarding the thres value, based on the AUROC maybe?
    output_thresholded = (
        output_test > best_threshold
    ).int()  # values > 0.1 → 1, else → 0

    # calculate accuracy
    count = 0
    for i in range(label_test.size(dim=0)):
        if output_thresholded[i] == label_test[i]:
            count = count + 1

    ### print results
    print(f"Testing accuracy is {100 * (count / label_test.size(dim=0))}%")

    ### saving results
    df_test_acc = pd.DataFrame({"Testing Acc": [count / label_test.size(dim=0)]})
    df_test_acc.to_csv(result_path + "Testing_best_Acc.csv", index=False)

    # Calculate True Positives (TP) and False Positives (FP)
    TP = (
        (label_test == 1) & (output_thresholded == 1)
    ).sum()  # Both label and prediction are 1
    print(f"TP: {TP}")
    FN = (
        (label_test == 1) & (output_thresholded == 0)
    ).sum()  # Label is 1, prediction is 0
    print(f"FN: {FN}")
    FP = (
        (label_test == 0) & (output_thresholded == 1)
    ).sum()  # Label is 0, prediction is 1
    print(f"FP: {FP}")
    TN = (
        (label_test == 0) & (output_thresholded == 0)
    ).sum()  # Both label and prediction are 0
    print(f"TN: {TN}")

    # Calculate Precision/PPV
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0  # Avoid division by zero
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0  # Avoid division by zero
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )  # Avoid division by zero

    print(f"Precision/PPV: {precision:.2f}")
    print(f"F1 score: {f1_score:.2f}")

    ### saving results
    df_test_acc = pd.DataFrame({"Testing Pre": [precision]})
    df_test_acc.to_csv(result_path + "Testing_best_Pre.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--model_type",
        "-m",
        default="Transformer",
        choices=["RNN", "LSTM", "Transformer", "BioClinicalBERT"],
        type=str,
    )
    parser.add_argument(
        "--data_type",
        "-d",
        default="EHR",
        choices=["EHR", "Report", "EHRAndReport"],
        type=str,
    )

    parser.add_argument(
        "--data_dir", default=DATA_DIR, type=str, help="Path to data directory root"
    )
    parser.add_argument(
        "--checkpoints_dir",
        default=CHECKPOINTS_DIR,
        type=str,
        help="Path to checkpoints directory root",
    )

    parser.add_argument(
        "--batch_size", "--bz", default=64, type=int, help="Mini-batch size"
    )
    parser.add_argument(
        "--hidden_size", "--hz", default=32, type=int, help="Hidden size (RNN/LSTM)"
    )
    parser.add_argument(
        "--num_layers", "-l", default=3, type=int, help="Number of layers (RNN/LSTM)"
    )
    parser.add_argument(
        "--num_epoch", "-e", default=1, type=int, help="Number of epochs"
    )
    parser.add_argument("--lr", default=5e-3, type=float, help="Learning rate")

    parser.add_argument(
        "--trans_input_dim",
        default=512,
        type=int,
        help="Custom transformer input dimensionality",
    )
    parser.add_argument(
        "--trans_n_heads",
        default=8,
        type=int,
        help="Transformer number of attention heads",
    )
    parser.add_argument(
        "--trans_ff_dim",
        default=2048,
        type=int,
        help="Transformer feedforward network dimensionality",
    )
    parser.add_argument(
        "--trans_dropout",
        default=0.1,
        type=float,
        help="Transformer dropout probability",
    )
    parser.add_argument(
        "--trans_num_layers", default=12, type=int, help="Transformer number of layers"
    )

    parser.add_argument(
        "--output_l1_dim",
        default=256,
        type=int,
        help="Final FF layer(1) dimensionality",
    )
    parser.add_argument(
        "--output_l2_dim",
        default=128,
        type=int,
        help="Final FF layer(2) dimensionality",
    )
    parser.add_argument(
        "--output_l3_dim",
        default=1,
        type=int,
        help="Classification layer dimensionality",
    )

    parser.add_argument(
        "--bert_use_temporal_conv",
        action="store_true",
        help="BioClinicalBERT: Whether to use a temporal convolution.",
    )

    args = parser.parse_args()

    main(**vars(args))
