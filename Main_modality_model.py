# import functions
# Standard Library
import copy
import os

# define fixed seed
import random

# Third-Party
import Model_n_Dataset

# Scientific Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.optim import lr_scheduler
# from sklearn.metrics.pairwise import cosine_similarity
# from torchvision import datasets, models, transforms, utils

seed_cus = 3407


def set_seed(seed_cus):
    random.seed(seed_cus)
    np.random.seed(seed_cus)
    torch.manual_seed(seed_cus)
    torch.cuda.manual_seed(seed_cus)
    torch.cuda.manual_seed_all(seed_cus)


def worker_init_fn(worker_id):
    seed = seed_cus  # Or any constant
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


g = torch.Generator()
g.manual_seed(seed_cus)

# set parameters
model = "Transformer"
data = "EHR"
batch_size = 64

hidden_size = 32
num_layers = 3

trans_input_dim = 512
trans_n_heads = 8
trans_ff_dim = 2048
trans_dropout = 0.1
trans_num_layers = 12
trans_max_len = 512

output_l1_dim = 256
output_l2_dim = 128
output_l3_dim = 1
num_epoch = 1
lr = 0.0005
best_auroc = -0.1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./data/"

if __name__ == "__main__":

    set_seed(seed_cus)
    # define train and test dataset
    if data == "EHR":
        Dataset = Model_n_Dataset.EHRDataset
    elif data == "Report":
        Dataset = Model_n_Dataset.ReportDataset
    else:
        Dataset = None

    assert Dataset is not None

    train_dataset = Dataset(os.path.join(DATA_DIR, "train_with_raw_report"))
    test_dataset = Dataset(os.path.join(DATA_DIR, "test_with_raw_report"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ### model saving
    # change path accroding to the model structure (Encoder/Discriminator/Scheduler) and data (partial or full)
    model_path = f"FullData_Model/Dataset_{str(data)}_Model_{str(model)}_epoch_{str(num_epoch)}_CosLR_lr_{str(lr)}_seed{str(seed_cus)}/"
    isExist = os.path.exists(model_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(model_path)
        print("The new model directory is created!")

    ### result saving
    # change path accroding to the model structure (Encoder/Discriminator/Scheduler) and data (partial or full)
    result_path = f"FullData_Result/Dataset_{str(data)}_Model_{str(model)}_epoch_{str(num_epoch)}_CosLR_lr_{str(lr)}_seed{str(seed_cus)}/"
    isExist = os.path.exists(result_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(result_path)
        print("The new result directory is created!")

    # set criterion
    criterion = nn.BCELoss()

    # define models
    if model == "RNN":
        Model_list = [
            Model_n_Dataset.CustomRNN(
                train_dataset[0][0].shape[-1], hidden_size, num_layers
            ),
            Model_n_Dataset.Model_head_3_layers(
                train_dataset[0][0].shape[-1],
                output_l1_dim,
                output_l2_dim,
                output_l3_dim,
            ),
        ]
        Model_name = ["Model_RNN", "Model_head_linear"]
    elif model == "LSTM":
        Model_list = [
            Model_n_Dataset.CustomLSTM(
                train_dataset[0][0].shape[-1], hidden_size, num_layers
            ),
            Model_n_Dataset.Model_head_3_layers(
                train_dataset[0][0].shape[-1],
                output_l1_dim,
                output_l2_dim,
                output_l3_dim,
            ),
        ]
        Model_name = ["Model_LSTM", "Model_head_linear"]
    elif model == "Transformer":
        Model_list = [
            Model_n_Dataset.Model_linear_1_layer(
                train_dataset[0][0].shape[-1], trans_input_dim
            ),
            Model_n_Dataset.CustomTransformer(
                trans_input_dim,
                trans_n_heads,
                trans_ff_dim,
                trans_num_layers,
                trans_max_len,
                trans_dropout,
            ),
            Model_n_Dataset.Model_head_1_layer(trans_input_dim, output_l3_dim),
        ]
        Model_name = ["Model_linear", "Model_Transformer", "Model_head_linear"]

    for model in Model_list:
        model.to(device)

    grad_params = []
    optimizers = []
    schedulers = []
    for model in Model_list:
        ### scheduler for model
        params = [param for param in model.parameters() if param.requires_grad]
        grad_params.append(params)
        optimizer = torch.optim.Adam(
            params, lr=lr, betas=(0.5, 0.99), weight_decay=1e-4
        )
        optimizers.append(optimizer)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epoch
        )
        schedulers.append(scheduler)

    epoch_all = []
    auroc_all = []

    # training loop
    for epoch in range(num_epoch):
        # start training
        print("start training")
        for model in Model_list:
            model.train()
        epoch_loss = 0.0

        for i, (ehr, label) in enumerate(train_loader):
            num_batches = len(train_loader)
            for opt in optimizers:
                opt.zero_grad()

            ehr = ehr.to(device)
            label = label.to(device).unsqueeze(1)

            data = ehr
            for model in Model_list:
                output = model(data)
                data = output
            loss = criterion(output, label)

            loss.backward()
            for opt in optimizers:
                opt.step()

            current_loss = loss.item()
            epoch_loss += current_loss

            print(f"Epoch: {epoch + 1}, Batch: {i}, Loss: {current_loss}")

        avg_epoch_loss = epoch_loss / num_batches
        print(f"Number of epoch completed: {epoch + 1}, Epoch Loss: {avg_epoch_loss}")

        for scheduler in schedulers:
            scheduler.step()

        output_test = []
        label_test = []

        for model in Model_list:
            model.eval()

        print("start testing")
        for i, (ehr, label) in enumerate(test_loader):
            ehr = ehr.to(device)
            label = label.to(device).unsqueeze(1)

            data = ehr
            for model in Model_list:
                output = model(data)
                data = output
            loss = criterion(output, label)

            current_loss = loss.item()
            epoch_loss += current_loss

            output_test.append(output.detach().cpu())
            label_test.append(label.detach().cpu())

        output_test = torch.cat(output_test)
        label_test = torch.cat(label_test)

        # Calculate AUROC or AUPRC
        auroc = roc_auc_score(label_test, output_test)
        # auprc = average_precision_score(label_test, output_test)

        # saving results
        epoch_all.append(epoch)
        auroc_all.append(auroc)
        df_test_auroc = pd.DataFrame({"Epoch": epoch_all, "Testing AUROC": auroc_all})
        df_test_auroc.to_csv(result_path + "Testing_all_AUROC.csv", index=False)
        # print results
        print(f"Testing AUROC is {100 * auroc}%")
        if auroc > best_auroc:
            best_auroc = auroc
            best_models = []
            for model in Model_list:
                best_model = copy.deepcopy(model.state_dict())
                best_models.append(best_model)

    ### loading best model weights
    for i, model in enumerate(Model_list):
        model.load_state_dict(best_models[i])
    ### saving trained model
    print("...saving model...")
    for model, model_name in zip(Model_list, Model_name):
        torch.save(model.state_dict(), model_path + f"{model_name}.pt")

    # define testing process after training completion and model saving
    print("start final testing")
    count = 0
    output_test = []
    label_test = []

    ### load best model
    for model, model_name in zip(Model_list, Model_name):
        model.load_state_dict(torch.load(model_path + f"{model_name}.pt"))
    for model in Model_list:
        model.eval()

    ### testing
    for i, (ehr, label) in enumerate(test_loader):
        ehr = ehr.to(device)
        label = label.to(device).unsqueeze(1)

        data = ehr
        for model in Model_list:
            output = model(data)
            data = output

        # append all outputs
        output_test.append(output.detach().cpu())

        # append all labels
        label_test.append(label.detach().cpu())

    # create list for all outputs
    output_test = torch.cat(output_test)

    # create list for all label
    label_test = torch.cat(label_test)

    # calculate AUROC and AUPRC
    auroc = roc_auc_score(label_test, output_test)
    auprc = average_precision_score(label_test, output_test)

    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")

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
    print(TP)
    FN = (
        (label_test == 1) & (output_thresholded == 0)
    ).sum()  # Label is 1, prediction is 0
    print(FN)
    FP = (
        (label_test == 0) & (output_thresholded == 1)
    ).sum()  # Label is 0, prediction is 1
    print(FP)
    TN = (
        (label_test == 0) & (output_thresholded == 0)
    ).sum()  # Both label and prediction are 0
    print(TN)

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
