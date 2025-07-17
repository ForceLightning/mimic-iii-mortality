# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 08:20:55 2025

This is the main script for running training and inference

@author: Yang
"""

# %% import packages

# Standard Library
import copy
import os

# define fixed seed
import random

# Third-Party
import Model_BCE

# Scientific Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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

# %% configurations


### input output dimension
linear1_input_dim_ehr = 76
linear1_input_dim_report = 768
linear1_hidden_dim = 512
linear1_output_dim = 512

linear2_input_dim_ehr = 512
linear2_input_dim_report = 512
linear2_hidden_dim = 512
linear2_output_dim = 512

Trans_input_dim = 512
Trans_n_heads = 8
Trans_ff_dim = 2048
Trans_dropout = 0.1
Trans_num_layers = 12
Trans_max_len = 512
Trans_seq_len = 48

cla_conversion_input_dim = 512
cla_conversion_hidden_dim_1 = 480
cla_conversion_hidden_dim_2 = 320
cla_conversion_output_dim = 256

cla_head_input_dim = 256
cla_head_hidden_dim_1 = 256
cla_head_hidden_dim_2 = 128
cla_head_output_dim = 25

### hyper-parameter
batch_size_cus = 64
epoch_max = 50
learning_rate = 0.0005
best_auroc = -0.1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_selection_loading = "last_48_hour"  # first_48_hour last_48_hour
# data_selection_saving = 'Acute_myocardial_nfarction_last_48_hour' # first_48_hour last_48_hour
# data_selection_saving = 'Cardiac_dysrhythmias_last_48_hour' # first_48_hour last_48_hour
# data_selection_saving = 'Congestive_heart_failure_nonhypertensive_last_48_hour' # first_48_hour last_48_hour
# data_selection_saving = 'Coronary_atherosclerosis_and_other_heart_disease_last_48_hour' # first_48_hour last_48_hour
# data_selection_saving = 'Conduction_disorders_last_48_hour' # first_48_hour last_48_hour
data_selection_saving = "All_classes_last_48_hour"  # first_48_hour last_48_hour

# %% define and create dataset (using processed MIMIC III data)


class EHRReportDataset(Dataset):
    def __init__(self, root_dir):
        self.ehr_dir = os.path.join(root_dir, "EHR/")
        self.report_dir = os.path.join(root_dir, "Report/")
        label_path = os.path.join(root_dir, "labels.csv")

        self.labels_df = pd.read_csv(label_path, index_col=False)
        self.stay_ids = self.labels_df["stay"].astype(str)
        self.labels = self.labels_df.iloc[:, 2:].astype(int)
        # self.labels = self.labels_df['Acute myocardial infarction'].astype(int)
        # self.labels = self.labels_df['Cardiac dysrhythmias'].astype(int)
        # self.labels = self.labels_df['Congestive heart failure; nonhypertensive'].astype(int)
        # self.labels = self.labels_df['Coronary atherosclerosis and other heart disease'].astype(int)
        # self.labels = self.labels_df['Conduction disorders'].astype(int)
        # here we only pick all 5 classes as multilabel classification

        # self.stay_ids = self.labels_df['stay'].astype(str).tolist()
        # self.labels = self.labels_df.set_index('stay')['y_true'].to_dict()

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        stay_id = self.stay_ids[idx]

        ehr_path = os.path.join(self.ehr_dir, f"{stay_id}")
        report_path = os.path.join(self.report_dir, f"{stay_id}")

        # ehr_df = pd.read_csv(ehr_path,  index_col=False)
        # report_df = pd.read_csv(report_path,  index_col=False)
        # use these instead if running into errors on Linux
        ehr_df = (
            pd.read_csv(ehr_path, index_col=False, encoding="utf-8")
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
        )
        report_df = (
            pd.read_csv(report_path, index_col=False, encoding="utf-8")
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
        )

        # Convert to tensors
        ehr_tensor = torch.tensor(ehr_df.values, dtype=torch.float32)
        report_tensor = torch.tensor(report_df.values, dtype=torch.float32)

        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float)

        return ehr_tensor, report_tensor, label


# Usage
train_dataset = EHRReportDataset(
    f"/home/user/Documents/AI_FH_mimic3_v2/1_data_processing/processed_data/phenotyping_{data_selection_loading}/train/"
)
test_dataset = EHRReportDataset(
    f"/home/user/Documents/AI_FH_mimic3_v2/1_data_processing/processed_data/phenotyping_{data_selection_loading}/test/"
)

train_loader = DataLoader(train_dataset, batch_size=batch_size_cus, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size_cus, shuffle=False)

# # Synthetic dataset
# class SyntheticDataset(Dataset):
#     def __init__(self, num_samples=1000):
#         super().__init__()
#         self.num_samples = num_samples

#         # Initialize tensors
#         self.x1 = torch.randn(num_samples, 48, 76)
#         self.x2 = torch.randn(num_samples, 48, 768)

#         # Generate labels
#         # For high correlation, we create a hidden signal
#         # e.g., sum over certain features, thresholded
#         hidden_signal1 = self.x1[:, :, :5].mean(dim=(1, 2))  # (batch_size,)
#         hidden_signal2 = self.x2[:, :, :20].mean(dim=(1, 2))  # (batch_size,)

#         # Combine signals
#         combined_signal = hidden_signal1 + hidden_signal2

#         # Create binary labels
#         self.labels = (combined_signal > combined_signal.median()).float()

#         # Optional: Slightly amplify the "class patterns"
#         self.x1[self.labels == 1] += 1.0
#         self.x2[self.labels == 1] += 0.5

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         return self.x1[idx], self.x2[idx], self.labels[idx]

# # Create the full dataset
# full_dataset = SyntheticDataset(num_samples=5000)

# # Train-test split
# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# # Example DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=batch_size_cus, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size_cus, shuffle=False)

# # # Example usage
# # for batch_x1, batch_x2, batch_labels in train_loader:
# #     print(batch_x1.shape)  # [32, 48, 76]
# #     print(batch_x2.shape)  # [32, 48, 768]
# #     print(batch_labels.shape)  # [32]
# #     break


# %% baseline selection

### specify version name
version_name = "V5"  ## choices are ['V1', 'V2', 'V3', 'V4'， 'V5']

### specify encoder and classification head


if version_name == "V1":
    ehr_linear1_choice = "Linear_1_layer"
    ehr_linear2_choice = "Linear_1_layer"
    report_linear1_choice = "Linear_1_layer"
    report_linear2_choice = "Linear_1_layer"
    cla_conversion_choice = "MLP_conv_2_layer"
    cla_head_choice = "MLP_head_3_layer"  # or '1layer_MLP'

if version_name == "V2":
    ehr_linear1_choice = "Linear_1_layer"
    ehr_linear2_choice = "Linear_1_layer"
    report_linear1_choice = "Linear_1_layer"
    report_linear2_choice = "Linear_1_layer"
    cla_conversion_choice = "MLP_conv_2_layer"
    cla_head_choice = "MLP_head_3_layer"  # or '1layer_MLP'


if version_name == "V3":
    ehr_linear1_choice = "Linear_1_layer"
    ehr_linear2_choice = "Trans_cross_fusion_layer_learnable"
    report_linear1_choice = "Linear_1_layer"
    report_linear2_choice = "Trans_cross_fusion_layer_learnable"
    cla_conversion_choice = "MLP_conv_2_layer"
    cla_head_choice = "MLP_head_3_layer"  # or '1layer_MLP'


if version_name == "V4":
    ehr_linear1_choice = "Linear_1_layer"
    # ehr_linear2_choice = 'Trans_cross_fusion_layer_learnable'
    report_linear1_choice = "Linear_1_layer"
    # report_linear2_choice = 'Trans_cross_fusion_layer_learnable'
    ehr_report_linear2_choice = "Trans_cross_fusion_layer_learnable"
    cla_conversion_choice = "MLP_conv_2_layer"
    cla_head_choice = "MLP_head_3_layer"  # or '1layer_MLP'

if version_name == "V5":
    ehr_linear1_choice = "Linear_1_layer"
    # ehr_linear2_choice = 'Trans_cross_fusion_layer_learnable'
    report_linear1_choice = "Linear_1_layer"
    # report_linear2_choice = 'Trans_cross_fusion_layer_learnable'
    ehr_report_linear2_choice = "Trans_cross_fusion_layer_learnable"
    cla_conversion_choice = "MLP_conv_2_layer"
    cla_head_choice = "MLP_head_3_layer"  # or '1layer_MLP'

# elif baseline_name == 'V1':
#     ehr_linear1_choice = 'Linear_1_layer'
#     ehr_linear2_choice = 'Linear_1_layer'
#     report_linear1_choice = 'MLP_conv_2_layer'
#     report_linear2_choice = 'MLP_conv_2_layer'
#     cla_conversion = 'MLP_2_layer'
#     classification_head = 'MLP_3_layer' # or '1layer_MLP'


# %% import models for various baselines


### linear structure
def model_linear(model_choice):
    if model_choice == "Linear_1_layer":
        return Model_BCE.MLP_1layer_Model
    elif model_choice == "Linear_2_layer":
        return Model_BCE.MLP_2layer_Model
    elif model_choice == "Linear_3_layer":
        return Model_BCE.MLP_3layer_Model


def model_fusion(model_choice):
    if model_choice == "Trans_cross_fusion_layer_learnable":
        return Model_BCE.DualTransformer_learnable
    # if model_choice == "Trans_cross_fusion_layer_sinusoidal":
    #     return Model.Transformer12Layer_sinusoidal


### classification conversion and head structure
def model_con(model_choice):
    if model_choice == "MLP_conv_2_layer":
        return Model_BCE.Classification_conv_2layer_model
    elif model_choice == "MLP_conv_3_layer":
        return Model_BCE.Classification_conv_3layer_model


def model_cla(model_choice):
    if model_choice == "MLP_head_1_layer":
        return Model_BCE.Classification_head_1layer_model
    elif model_choice == "MLP_head_2_layer":
        return Model_BCE.Classification_head_2layer_model
    elif model_choice == "MLP_head_3_layer":
        return Model_BCE.Classification_head_3layer_model


### load model
model_ehr_linear_1 = model_linear(ehr_linear1_choice)
# model_ehr_linear_2 = model_fusion(ehr_linear2_choice)

model_report_linear_1 = model_linear(report_linear1_choice)
# model_report_linear_2 = model_fusion(report_linear2_choice)

model_ehr_report_linear_2 = model_fusion(ehr_report_linear2_choice)

model_cla_conver = model_con(cla_conversion_choice)
model_cla_head = model_cla(cla_head_choice)

### specify model parameters for model
if ehr_linear1_choice == "Linear_1_layer":
    Model_EHR_L1 = model_ehr_linear_1(linear1_input_dim_ehr, linear1_output_dim)
elif ehr_linear1_choice == "Linear_2_layer":
    Model_EHR_L1 = model_ehr_linear_1(
        linear1_input_dim_ehr, linear1_hidden_dim, linear1_output_dim
    )
elif ehr_linear1_choice == "Linear_3_layer":
    Model_EHR_L1 = model_ehr_linear_1(
        linear1_input_dim_ehr, linear1_hidden_dim, linear1_output_dim
    )
else:
    print("Model_EHR_L1 is not defined")

# if ehr_linear2_choice == 'Linear_1_layer' :
#     Model_EHR_L2 = model_ehr_linear_2(linear2_input_dim_ehr, linear2_output_dim)
# elif ehr_linear2_choice == 'Linear_2_layer' :
#     Model_EHR_L2 = model_ehr_linear_2(linear2_input_dim_ehr, linear2_hidden_dim, linear2_output_dim)
# elif ehr_linear2_choice == 'Linear_3_layer' :
#     Model_EHR_L2 = model_ehr_linear_2(linear2_input_dim_ehr, linear2_hidden_dim, linear2_output_dim)
# elif ehr_linear2_choice == 'Trans_cross_fusion_layer_learnable' :
#     Model_EHR_L2 = model_ehr_linear_2(Trans_input_dim, Trans_n_heads, Trans_ff_dim, Trans_num_layers, Trans_max_len, Trans_dropout)
# elif ehr_linear2_choice == 'Trans_cross_fusion_layer_sinusoidal' :
#     Model_EHR_L2 = model_ehr_linear_2(Trans_input_dim, Trans_n_heads, Trans_ff_dim, Trans_num_layers, Trans_max_len, Trans_dropout, Trans_seq_len)
# else:
#     print('Model_EHR_L2 is not defined')

if report_linear1_choice == "Linear_1_layer":
    Model_Report_L1 = model_report_linear_1(
        linear1_input_dim_report, linear1_output_dim
    )
elif report_linear1_choice == "Linear_2_layer":
    Model_Report_L1 = model_report_linear_1(
        linear1_input_dim_report, linear1_hidden_dim, linear1_output_dim
    )
elif report_linear1_choice == "Linear_3_layer":
    Model_Report_L1 = model_report_linear_1(
        linear1_input_dim_report, linear1_hidden_dim, linear1_output_dim
    )
else:
    print("Model_Report_L1 is not defined")

# if report_linear2_choice == 'Linear_1_layer' :
#     Model_Report_L2 = model_report_linear_2(linear2_input_dim_report, linear2_output_dim)
# elif report_linear2_choice == 'Linear_2_layer' :
#     Model_Report_L2 = model_report_linear_2(linear2_input_dim_report, linear2_hidden_dim, linear2_output_dim)
# elif report_linear2_choice == 'Linear_3_layer' :
#     Model_Report_L2 = model_report_linear_2(linear2_input_dim_report, linear2_hidden_dim, linear2_output_dim)
# elif report_linear2_choice == 'Trans_cross_fusion_layer_learnable' :
#     Model_Report_L2 = model_report_linear_2(Trans_input_dim, Trans_n_heads, Trans_ff_dim, Trans_num_layers, Trans_max_len, Trans_dropout)
# elif report_linear2_choice == 'Trans_cross_fusion_layer_sinusoidal' :
#     Model_Report_L2 = model_report_linear_2(Trans_input_dim, Trans_n_heads, Trans_ff_dim, Trans_num_layers, Trans_max_len, Trans_dropout, Trans_seq_len)
# else:
#     print('Model_Report_L2 is not defined')

if ehr_report_linear2_choice == "Trans_cross_fusion_layer_learnable":
    Model_EHR_Report_L2 = model_ehr_report_linear_2(
        Trans_input_dim,
        Trans_n_heads,
        Trans_ff_dim,
        Trans_num_layers,
        Trans_max_len,
        Trans_dropout,
    )
else:
    print("Model_Report_L2 is not defined")

if cla_conversion_choice == "MLP_conv_2_layer":
    Model_Cls_Conv = model_cla_conver(
        cla_conversion_input_dim, cla_conversion_hidden_dim_1, cla_conversion_output_dim
    )
elif cla_conversion_choice == "MLP_conv_3_layer":
    Model_Cls_Conv = model_cla_conver(
        cla_conversion_input_dim,
        cla_conversion_hidden_dim_1,
        cla_conversion_hidden_dim_2,
        cla_conversion_output_dim,
    )
else:
    print("Model_Cls_Conv is not defined")

if cla_head_choice == "MLP_head_2_layer":
    Model_Cls_Head = model_cla_head(
        cla_head_input_dim, cla_head_hidden_dim_2, cla_head_output_dim
    )
elif cla_head_choice == "MLP_head_3_layer":
    Model_Cls_Head = model_cla_head(
        cla_head_input_dim,
        cla_head_hidden_dim_1,
        cla_head_hidden_dim_2,
        cla_head_output_dim,
    )
else:
    print("Model_Cls_Conv is not defined")


# ### pass model to GPU device
Model_EHR_L1.to(device)
# Model_EHR_L2.to(device)
Model_Report_L1.to(device)
# Model_Report_L2.to(device)
Model_EHR_Report_L2.to(device)
Model_Cls_Conv.to(device)
Model_Cls_Head.to(device)


# %% define model path and result path

### model saving
# change path accroding to the model structure (Encoder/Discriminator/Scheduler) and data (partial or full)
model_path = f"FullData_Model/version_{version_name}_Train_v1_phenotyping_{data_selection_saving}_multilabel_epoch_{str(epoch_max)}_CosLR_lr_{str(learning_rate)}_seed{str(seed_cus)}/"
isExist = os.path.exists(model_path)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(model_path)
    print("The new model directory is created!")

### result saving
# change path accroding to the model structure (Encoder/Discriminator/Scheduler) and data (partial or full)
result_path = f"FullData_Result/version_{version_name}_Train_v1_phenotyping_{data_selection_saving}_multilabel_epoch_{str(epoch_max)}_CosLR_lr_{str(learning_rate)}_seed{str(seed_cus)}/"
isExist = os.path.exists(result_path)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(result_path)
    print("The new result directory is created!")


# %% define loss and optimizer/scheduler

### loss
criterion = nn.BCELoss()

### scheduler for model1
grad_params_1 = [param for param in Model_EHR_L1.parameters() if param.requires_grad]
optimizer_1 = torch.optim.Adam(
    grad_params_1, lr=learning_rate, betas=(0.5, 0.99), weight_decay=1e-4
)
scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=epoch_max)

### scheduler for model2
grad_params_2 = [param for param in Model_Report_L1.parameters() if param.requires_grad]
optimizer_2 = torch.optim.Adam(
    grad_params_2, lr=learning_rate, betas=(0.5, 0.99), weight_decay=1e-4
)
scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=epoch_max)

### scheduler for model3
grad_params_3 = [
    param for param in Model_EHR_Report_L2.parameters() if param.requires_grad
]
optimizer_3 = torch.optim.Adam(
    grad_params_3, lr=learning_rate, betas=(0.5, 0.99), weight_decay=1e-4
)
scheduler_3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_3, T_max=epoch_max)

### scheduler for model4
grad_params_4 = [param for param in Model_Cls_Conv.parameters() if param.requires_grad]
optimizer_4 = torch.optim.Adam(
    grad_params_4, lr=learning_rate, betas=(0.5, 0.99), weight_decay=1e-4
)
scheduler_4 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_4, T_max=epoch_max)

### scheduler for model5
grad_params_5 = [param for param in Model_Cls_Head.parameters() if param.requires_grad]
optimizer_5 = torch.optim.Adam(
    grad_params_5, lr=learning_rate, betas=(0.5, 0.99), weight_decay=1e-4
)
scheduler_5 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_5, T_max=epoch_max)


# %% define training process
### log account
auroc_all = []
epoch_all = []

### training
for epoch in range(epoch_max):
    # start training
    print("start training")

    # ### pass model to GPU device
    Model_EHR_L1.train()
    # Model_EHR_L2.train()
    Model_Report_L1.train()
    # Model_Report_L2.train()
    Model_EHR_Report_L2.train()
    Model_Cls_Conv.train()
    Model_Cls_Head.train()

    loss_value = 0
    running_loss = 0.0

    for i, (ehr, report, labels) in enumerate(train_loader):

        num_batches = len(train_loader)
        # print(i)

        # zero optimizer
        optimizers = [optimizer_1, optimizer_2, optimizer_3, optimizer_4, optimizer_5]
        for opt in optimizers:
            opt.zero_grad()

        # print(ehr_data)
        # print(report_data)

        # pass the data to GPU
        ehr_data = ehr.to(device)
        report_data = report.to(device)
        label = labels.to(device)
        # print(ehr_data.shape)
        # print(report_data.x.shape)

        # 1st linear conversion, 76/768 to 512
        ehr_output_conv = Model_EHR_L1(ehr_data)
        report_output_conv = Model_Report_L1(report_data)

        # 2nd nonlinear conversion as cross-model fusion using Transformer, 48-hour sequence reduced to 1 with [CLS]
        ehr_output_fusion, report_output_fusion = Model_EHR_Report_L2(
            ehr_output_conv, report_output_conv
        )

        # dimension conversion for classification, 512 to 256
        ehr_output_cla = Model_Cls_Conv(ehr_output_fusion)
        report_output_cla = Model_Cls_Conv(report_output_fusion)

        # classification head
        final_output = Model_Cls_Head(ehr_output_cla, report_output_cla)

        # calcuate loss
        loss = criterion(final_output, label)  # could also use report.y

        # back propogation
        loss.backward()
        for opt in optimizers:
            opt.step()

        current_loss = loss.item()
        running_loss += current_loss
        print(f"Epoch: {epoch}, Batch: {i}, Loss: {current_loss}")

    epoch_loss = running_loss / num_batches
    print(f"Number of Epoch Completed: {epoch}, Epoch Loss: {epoch_loss}")

    # update learning rate
    schedulers = [scheduler_1, scheduler_2, scheduler_3, scheduler_4, scheduler_5]
    for scheduler in schedulers:
        scheduler.step()

    # testing within every training epoch
    print("start testing")
    # count = 0
    output_test = []
    label_test = []
    Model_EHR_L1.eval()
    # Model_EHR_L2.eval()
    Model_Report_L1.eval()
    # Model_Report_L2.eval()
    Model_EHR_Report_L2.eval()
    Model_Cls_Conv.eval()
    Model_Cls_Head.eval()
    for i, (ehr, report, labels) in enumerate(test_loader):

        # pass the data to GPU
        ehr_data = ehr.to(device)
        report_data = report.to(device)
        label = labels.to(device)
        # print(ehr_data.shape)
        # print(report_data.x.shape)

        # 1st linear conversion, 76/768 to 512
        ehr_output_conv = Model_EHR_L1(ehr_data)
        report_output_conv = Model_Report_L1(report_data)

        # 2nd nonlinear conversion as cross-model fusion using Transformer, 48-hour sequence reduced to 1 with [CLS]
        ehr_output_fusion, report_output_fusion = Model_EHR_Report_L2(
            ehr_output_conv, report_output_conv
        )

        # dimension conversion for classification, 512 to 256
        ehr_output_cla = Model_Cls_Conv(ehr_output_fusion)
        report_output_cla = Model_Cls_Conv(report_output_fusion)

        # classification head
        final_output = Model_Cls_Head(ehr_output_cla, report_output_cla)

        # append all outputs
        output_test.append(final_output.detach().cpu())

        # append all labels
        label_test.append(label.detach().cpu())

    # create list for all outputs
    output_test = torch.cat(output_test)
    # print(output.shape)

    # select class with greater probability
    # output_test = torch.argmax(output_test, -1)
    # print(output.shape)

    # create list for all label
    label_test = torch.cat(label_test)
    # print(label.shape)

    # Calculate AUROC or AUPRC
    auroc = roc_auc_score(label_test, output_test, average="micro")
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
        best_model1_wts = copy.deepcopy(Model_EHR_L1.state_dict())
        best_model2_wts = copy.deepcopy(Model_Report_L1.state_dict())
        best_model3_wts = copy.deepcopy(Model_EHR_Report_L2.state_dict())
        best_model4_wts = copy.deepcopy(Model_Cls_Conv.state_dict())
        best_model5_wts = copy.deepcopy(Model_Cls_Head.state_dict())

### loading best model weights
Model_EHR_L1.load_state_dict(best_model1_wts)
Model_Report_L1.load_state_dict(best_model2_wts)
Model_EHR_Report_L2.load_state_dict(best_model3_wts)
Model_Cls_Conv.load_state_dict(best_model4_wts)
Model_Cls_Head.load_state_dict(best_model5_wts)
### saving trained model
print("...saving model...")
torch.save(Model_EHR_L1.state_dict(), model_path + "Train_V1_Model_EHR_L1.pt")
torch.save(Model_Report_L1.state_dict(), model_path + "Train_V1_Model_Report_L1.pt")
torch.save(
    Model_EHR_Report_L2.state_dict(), model_path + "Train_V1_Model_EHR_Report_L2.pt"
)
torch.save(Model_Cls_Conv.state_dict(), model_path + "Train_V1_Model_Cls_Conv.pt")
torch.save(Model_Cls_Head.state_dict(), model_path + "Train_V1_Model_Cls_Head.pt")


# %% define testing process after training completion and model saving
print("start final testing")
count = 0
output = []
label_test = []

### load best model
Model_EHR_L1.load_state_dict(torch.load(model_path + "Train_V1_Model_EHR_L1.pt"))
Model_Report_L1.load_state_dict(torch.load(model_path + "Train_V1_Model_Report_L1.pt"))
Model_EHR_Report_L2.load_state_dict(
    torch.load(model_path + "Train_V1_Model_EHR_Report_L2.pt")
)
Model_Cls_Conv.load_state_dict(torch.load(model_path + "Train_V1_Model_Cls_Conv.pt"))
Model_Cls_Head.load_state_dict(torch.load(model_path + "Train_V1_Model_Cls_Head.pt"))
Model_EHR_L1.eval()
Model_Report_L1.eval()
Model_EHR_Report_L2.eval()
Model_Cls_Conv.eval()
Model_Cls_Head.eval()

### testing
for i, (ehr, report, labels) in enumerate(test_loader):

    # pass the data to GPU
    ehr_data = ehr.to(device)
    report_data = report.to(device)
    label = labels.to(device)
    # print(ehr_data.shape)
    # print(report_data.x.shape)

    # 1st linear conversion, 76/768 to 512
    ehr_output_conv = Model_EHR_L1(ehr_data)
    report_output_conv = Model_Report_L1(report_data)

    # 2nd nonlinear conversion as cross-model fusion using Transformer, 48-hour sequence reduced to 1 with [CLS]
    ehr_output_fusion, report_output_fusion = Model_EHR_Report_L2(
        ehr_output_conv, report_output_conv
    )

    # dimension conversion for classification, 512 to 256
    ehr_output_cla = Model_Cls_Conv(ehr_output_fusion)
    report_output_cla = Model_Cls_Conv(report_output_fusion)

    # classification head
    final_output = Model_Cls_Head(ehr_output_cla, report_output_cla)

    # append all outputs
    output.append(final_output.detach().cpu())

    # append all labels
    label_test.append(label.detach().cpu())

# create list for all outputs
output = torch.cat(output)
# print(output.shape)

# original output with probabilities
# output_auc = output
# select class with greater probability
# output = torch.argmax(output, -1)
# print(output.shape)

# create list for all label
label_test = torch.cat(label_test)
# print(label_test.shape)


# calculate AUROC and AUPRC
auroc = roc_auc_score(label_test, output, average="micro")
auprc = average_precision_score(label_test, output, average="micro")

print(f"micro AUROC: {auroc:.4f}")
print(f"micro AUPRC: {auprc:.4f}")

### saving results
df_test_acc = pd.DataFrame({"Testing micro AUROC": [auroc]})
df_test_acc.to_csv(result_path + "Testing_best_micro_AUROC.csv", index=False)

### saving results
df_test_acc = pd.DataFrame({"Testing micro AUPRC": [auprc]})
df_test_acc.to_csv(result_path + "Testing_best_micro_AUPRC.csv", index=False)


# To find the cut-off (threshold) value that gives the best classification performance from an AUROC for every class
def find_optimal_thresholds(label_test, output):
    n_classes = label_test.shape[1]
    thresholds = []

    for i in range(n_classes):
        fpr, tpr, thresh = roc_curve(label_test[:, i], output[:, i])
        youden_j = tpr - fpr
        best_thresh = thresh[np.argmax(youden_j)]
        thresholds.append(best_thresh)

    return torch.as_tensor(thresholds, device=output.device, dtype=output.dtype)


# apply threshold on output to calculate following metrics
optimal_thresholds = find_optimal_thresholds(label_test, output)
output_thresholded = (output >= optimal_thresholds).int()  # values > 0.1 → 1, else → 0


# Calculate precision score
precision = precision_score(label_test, output_thresholded, average="micro")
print(f"Testing micro precision is: {precision:.2f}")
### saving results
df_test_precision = pd.DataFrame({"Testing micro precision": [precision]})
df_test_precision.to_csv(result_path + "Testing_best_micro_precision.csv", index=False)

# Calculate recall score
recall = recall_score(label_test, output_thresholded, average="micro")
print(f"Testing micro recall is: {recall:.2f}")
### saving results
df_test_recall = pd.DataFrame({"Testing micro recall": [recall]})
df_test_recall.to_csv(result_path + "Testing_best_micro_recall.csv", index=False)

# Calculate F1 score
f1 = f1_score(label_test, output_thresholded, average="micro")
print(f"Testing micro F1 Score is: {f1:.2f}")
### saving results
df_test_f1 = pd.DataFrame({"Testing micro F1 Score": [f1]})
df_test_f1.to_csv(result_path + "Testing_best_micro_F1Score.csv", index=False)
