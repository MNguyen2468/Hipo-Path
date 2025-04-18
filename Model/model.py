import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
#from lifelines.utils import concordance_index  # For C-index calculation
def c_index(pred, ytime, yevent):
    '''Calculate concordance index to evaluate models.
    Input:
        pred: linear predictors from trained model.
        ytime: true survival time from load_data().
        yevent: true censoring status from load_data().
    Output:
        concordance_index: c-index (between 0 and 1).
    '''
    
    n_sample = len(ytime)
    ytime_indicator = R_set(ytime,len(pred))
    ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
    ###T_i is uncensored
    censor_idx = (yevent == 0)
    zeros = torch.zeros(n_sample)
    ytime_matrix[censor_idx, :] = zeros
    ###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
    pred_matrix = torch.zeros_like(ytime_matrix)
    for j in range(n_sample):
        for i in range(n_sample):
            if pred[i] < pred[j]:
                pred_matrix[j, i]  = 1
            elif pred[i] == pred[j]: 
                pred_matrix[j, i] = 0.5
    
    concord_matrix = pred_matrix.mul(ytime_matrix)
    ###numerator
    concord = torch.sum(concord_matrix)
    ###denominator
    epsilon = torch.sum(ytime_matrix)
    ###c-index = numerator/denominator
    concordance_index = torch.div(concord, epsilon)

    return(concordance_index)
def R_set(x,lenght):
    '''Create an indicator matrix of risk sets, where T_j >= T_i.
    Note that the input data have been sorted in descending order.
    Input:
        x: a PyTorch tensor that the number of rows is equal to the number of samples.
    Output:
        indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
    '''
    # x = x.cpu().detach().numpy()
    n_sample = lenght
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)

    return(indicator_matrix)
def cox_nll_loss(preds, times, events, l2_reg=1e-4):
    preds = preds.view(-1)
    times = times.view(-1)
    events = events.view(-1)

    order = torch.argsort(times, descending=True)
    preds, times, events = preds[order], times[order], events[order]

    risk = preds - torch.logcumsumexp(preds, dim=0)
    uncensored_likelihood = risk * events
    neg_log_likelihood = -torch.sum(uncensored_likelihood) / (events.sum() + 1e-8)

    l2_penalty = l2_reg * torch.mean(preds**2)

    return neg_log_likelihood + l2_penalty



# def cox_nll_loss(preds, times, events, l2_reg=1e-4):
#     order = torch.argsort(times, descending=True)
#     preds, times, events = preds[order], times[order], events[order]

#     risk_set = torch.cumsum(torch.exp(preds), dim=0)
#     log_likelihood = torch.sum(preds - torch.log(risk_set))

#     event_sum = torch.sum(events)
#     if event_sum == 0:  
#         return l2_reg * torch.sum(preds**2)  # Use L2 penalty as fallback

#     return -log_likelihood / event_sum + l2_reg * torch.sum(preds**2)
# -----------------------
# 1. Load Labels from TSV
# -----------------------
label_file = "/home/sai/Desktop/data_split/KM_Plot__Overall_(months).txt"  # Change to your actual file path
data_dir = "/home/sai/Desktop/data_split/hipo-results"
path_data_dir = "/home/sai/Desktop/data_split/combined_PCA_samples"

# Load TSV file
labels_df = pd.read_csv(label_file, sep="\t")

# Filter out rows where OS_MONTHS is NA
labels_df = labels_df.dropna(subset=["OS_MONTHS"])

# Extract Patient ID (column 2), OS_STATUS (column 3), and OS_MONTHS (column 4)
label_dict = dict(zip(labels_df.iloc[:, 1], zip(labels_df.iloc[:, 2], labels_df.iloc[:, 3])))

print(f"Loaded {len(label_dict)} valid labels.")

# -----------------------
# 2. Load `.npy` Files and Match Labels
# -----------------------
all_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
path_all_files = [f for f in os.listdir(path_data_dir) if f.endswith(".npy")]


# Match `.npy` files to labels using the first 12 characters
file_label_pairs = [(f, label_dict[f[:12]]) for f in all_files if f[:12] in label_dict]
path_file_label_pairs = [(f, label_dict[f[:12]]) for f in path_all_files if f[:12] in label_dict]

# Extract filenames from both lists
file_label_set = set((f,_) for f, _ in file_label_pairs)
path_file_label_set = set((f,_) for f, _ in path_file_label_pairs)

file_label_final = file_label_set.intersection(path_file_label_set)

file_label_final = list(file_label_final)
print(file_label_final[0])

# Verify that no targets are NaN
for file_name, (status, label) in file_label_pairs:
    if pd.isna(label):
        raise ValueError(f"NaN label found for file: {file_name}")

# -----------------------
# 3. Split Data Based on CSV Mapping
# -----------------------
# Load the dataset split CSV
split_df = pd.read_csv("/home/sai/Desktop/data_split/labeled_dataset.csv")

# Create a dictionary mapping filename -> train/test/validation
split_dict = dict(zip(split_df.iloc[:, 0], split_df.iloc[:, 1]))

# Create empty lists for each split
train_files, test_files, val_files = [], [], []
# Distribute files based on split_dict
for file_name, label in file_label_final:
    
    if file_name[:12] in split_dict:
        split_type = split_dict[file_name[:12]]
        file = file_name[:12] + ".npy"
        if split_type == "train":
            train_files.append((file, label))
        elif split_type == "test":
            test_files.append((file, label))
        elif split_type == "val":
            val_files.append((file, label))
        else:
            print(f"⚠️ Warning: Unrecognized split type '{split_type}' for {file_name}")
    else:
        print(f"⚠️ Warning: {file_name[:12]} not found in split CSV")
# Filter train_files to just deceased patients
train_files_dead_only = [item for item in train_files if item[1][0] != "0:LIVING"]
val_files_dead_only = [item for item in val_files if item[1][0] != "0:LIVING"]

# -----------------------
# 4. Verify Splits and Save
# -----------------------
print(f"✅ Train: {len(train_files)} samples")
print(f"✅ Test: {len(test_files)} samples")
print(f"✅ Validation: {len(val_files)} samples")
print(val_files)
# -----------------------
# 3. Custom Dataset Class
# -----------------------
class HipomapDataset(Dataset):
    def __init__(self, file_label_list, data_dir_hipomap,data_dir_pathcnn):
        self.file_label_list = file_label_list
        self.data_dir_hipomap = data_dir_hipomap
        self.data_dir_pathcnn = data_dir_pathcnn
    def __len__(self):
        return len(self.file_label_list)

    def __getitem__(self, idx):
        file_name, (status, label) = self.file_label_list[idx]
        file_path_hipo = os.path.join(self.data_dir_hipomap, file_name)
        file_path_pathcnn = os.path.join(self.data_dir_pathcnn, file_name)
        datahipo = np.load(file_path_hipo).astype(np.float32)
        datapathcnn = np.load(file_path_pathcnn).astype(np.float32)  # Load .npy file
        # Normalize Hipomap and PathCNN inputs
        datahipo = (datahipo - datahipo.mean()) / (datahipo.std() + 1e-8)
        datapathcnn = (datapathcnn - datapathcnn.mean()) / (datapathcnn.std() + 1e-8)

        X_hipo = torch.tensor(datahipo).unsqueeze(0) 
        X_pathcnn =  torch.tensor(datapathcnn).unsqueeze(0) # Add channel dimension
        y = torch.tensor(label, dtype=torch.float32)
        status = 0 if status == "0:LIVING" else 1  # Convert status to binary (0: living, 1: deceased)
        status = torch.tensor(status, dtype=torch.float32).unsqueeze(0) 
        return X_hipo, X_pathcnn, y, status

# Create datasets
train_dataset = HipomapDataset(train_files, data_dir,path_data_dir)

test_dataset = HipomapDataset(test_files, data_dir,path_data_dir)
val_dataset = HipomapDataset(val_files, data_dir,path_data_dir)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# -----------------------
# 4. Define CNN Model
# -----------------------
# class CNNRegression(nn.Module):
#     def __init__(self):
#         super(CNNRegression, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=2, padding=1),
#             nn.Tanh(),
#             nn.MaxPool2d(2, 2),
#             nn.Dropout(0.5),
#             nn.Conv2d(32, 32, kernel_size=2, padding=1),
#             nn.Tanh(),
#             nn.MaxPool2d((4, 4)),
#             nn.Dropout(0.5)  # Normalize output size
#         )
#         self.conv_path_layers = self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=2, padding=1),
#             nn.Tanh(),
#             nn.MaxPool2d(2, 2),
#             nn.Dropout(0.5),
#             nn.Conv2d(32, 32, kernel_size=2, padding=1),
#             nn.Tanh(),
#             nn.AdaptiveMaxPool2d((1, 1)),
#             nn.Dropout(0.5)  # Normalize output size
#         )
    
#         self.fc_layers_ = nn.Linear(64,1)
#         self.dropout_cnn = nn.Dropout(p=0.3)  # Dropout after CNN layers
#         self.dropout_fc = nn.Dropout(p=0.5) 


#     def forward(self, x_image,x_pathcnn):
#         x = self.conv_layers(x_image)
#         x1 =  self.conv_path_layers(x_pathcnn)
#         x = torch.flatten(x,start_dim=1)
#         x1 = torch.flatten(x1,start_dim=1)
#         x2 = torch.cat((x, x1), dim=1)
#         x2 = self.dropout_fc(x2)
#         x2 = self.fc_layers_(x2)
        

#         return x2
class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()

        # CNN block for Hipomap input
        self.conv_hipo = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 32, kernel_size=2, padding=1),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.5)
        )

        # CNN block for PathCNN input
        self.conv_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 32, kernel_size=2, padding=1),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.5)
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.output_scale = nn.Parameter(torch.tensor(10.0))
    def forward(self, x_hipo, x_pathcnn):
        x_hipo = self.conv_hipo(x_hipo)
        x_path = self.conv_path(x_pathcnn)

        x_hipo = x_hipo.view(x_hipo.size(0), -1)
        x_path = x_path.view(x_path.size(0), -1)

        x = torch.cat((x_hipo, x_path), dim=1)
        x = self.fc(x)
        #print("Output scale:", self.output_scale.item())
        
        return x * self.output_scale
# -----------------------
# 5. Train the Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNRegression().to(device)
model = torch.compile(model)  # Just after model definition

optimizer = optim.Adam(model.parameters(), lr=0.000001)
# Training loop
num_epochs = 500
val_c_index_total = []
train_c_index_total = []
best_val_c_index = -float("inf")
best_epoch = -1

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_targets = []
    all_outputs = []
    all_statuses = []

    for batch_idx, (inputhipomaps,inputpathcnns, targets, statuses) in enumerate(train_loader):
        inputhipomaps,inputpathcnns, targets, statuses = inputhipomaps.to(device),inputpathcnns.to(device), targets.to(device), statuses.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputhipomaps,inputpathcnns)
        #print("Num uncensored in training:", statuses.sum().item())
        #print("Train outputs:", outputs[:5].view(-1).tolist())


        # loss_fn = torch.nn.MSELoss()
        # loss = loss_fn(outputs.view(-1), targets.view(-1))

        loss = cox_nll_loss(outputs, targets, statuses)
        print(loss)
        loss.backward()
        #print("Grad norm (output_scale):", model.output_scale.grad.norm().item())
        optimizer.step()

        total_loss += loss.item()

        # Collect targets, outputs, and statuses for C-index calculation
        all_targets.extend(targets.cpu().detach().numpy())
        all_outputs.extend(outputs.cpu().detach().numpy())
        all_statuses.extend(statuses.cpu().detach().numpy())
    #print(all_targets, all_outputs, all_statuses)
    # Calculate C-index for training data
    train_c_index = c_index(all_outputs, all_targets, all_statuses)
    train_c_index_total.append(train_c_index)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Train C-Index: {train_c_index:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0
    val_targets = []
    val_outputs = []
    val_statuses = []

    with torch.no_grad():
        for inputhipomaps,inputpathcnns,targets, statuses in val_loader:
            inputhipomaps,inputpathcnns, targets, statuses = inputhipomaps.to(device),inputpathcnns.to(device), targets.to(device), statuses.to(device)

            outputs = model(inputhipomaps,inputpathcnns)

            # loss_fn = torch.nn.MSELoss()
            # loss = loss_fn(outputs.view(-1), targets.view(-1))
            loss = cox_nll_loss(outputs, targets, statuses)


            val_loss += loss.item()

            # Collect targets, outputs, and statuses for C-index calculation
            val_targets.extend(targets.view(-1).cpu().numpy())
            val_outputs.extend(outputs.view(-1).cpu().numpy())
            val_statuses.extend(statuses.view(-1).cpu().numpy())
    print("Val outputs sample:", val_outputs[:5])
    sorted_idx = np.argsort(val_outputs)
    print("Top risk scores:", np.array(val_outputs)[sorted_idx][-5:])
    print("Corresponding survival times:", np.array(val_targets)[sorted_idx][-5:])

    # Calculate C-index for validation data
    val_c_index = c_index( val_outputs,val_targets, val_statuses)
    val_c_index_total.append(val_c_index)
    best_c_index_list = []
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation C-Index: {val_c_index:.4f}")
    if val_c_index > best_val_c_index:
        best_val_c_index = val_c_index
        best_c_index_list.append(best_val_c_index)
        best_epoch = epoch
        
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ Best model saved at epoch {epoch+1} with Val C-Index: {val_c_index:.4f}")

print(f"best epoch: {best_epoch}")  
# -----------------------
# Final Testing Phase
# -----------------------
print("\n--- Final Testing ---")
model.load_state_dict(torch.load("best_model.pth", weights_only=True))

model.eval()
test_loss = 0
test_targets = []
test_outputs = []
test_statuses = []

with torch.no_grad():
    for inputhipomaps, inputpathcnns, targets, statuses in test_loader:
        inputhipomaps, inputpathcnns, targets, statuses = (
            inputhipomaps.to(device),
            inputpathcnns.to(device),
            targets.to(device),
            statuses.to(device),
        )

        outputs = model(inputhipomaps, inputpathcnns)
        # loss_fn = torch.nn.MSELoss()
        # loss = loss_fn(outputs.view(-1), targets.view(-1))
        loss = cox_nll_loss(outputs, targets, statuses)
        test_loss += loss.item()

        test_targets.extend(targets.cpu().numpy())
        test_outputs.extend(outputs.cpu().numpy())
        test_statuses.extend(statuses.cpu().numpy())

# Compute final test C-index
test_c_index = c_index( test_outputs,test_targets, test_statuses)
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
# Extract a simple feature (e.g., mean pixel value) from hipomap .npy files
baseline_features = []
baseline_times = []
baseline_events = []

for file_name, (status, time) in train_files:
    file_path = os.path.join(path_data_dir, file_name)
    data = np.load(file_path)
    feature_mean = np.mean(data)  # simple global feature
    baseline_features.append(feature_mean)
    baseline_times.append(time)
    baseline_events.append(0 if status == "0:LIVING" else 1)

import pandas as pd
# Create a DataFrame for lifelines
df_baseline = pd.DataFrame({
    'feature_mean': baseline_features,
    'duration': baseline_times,
    'event': baseline_events
})

# Fit the Cox model
cox_model = CoxPHFitter()
cox_model.fit(df_baseline, duration_col="duration", event_col="event")
cox_model.print_summary()
plt.plot(train_c_index_total, label='Train C-Index')
plt.plot(val_c_index_total, label='Val C-Index')
plt.plot(best_c_index_list, label = 'Best Val C-Index')
plt.xlabel('Epoch')
plt.ylabel('C-Index')
plt.legend()

plt.savefig("c_index_plot.png", dpi=300)  # Save as high-quality PNG

print(f"\nTest Loss: {test_loss / len(test_loader):.4f}, Test C-Index: {test_c_index:.4f}")
np.save('test_c_index.npy',test_c_index)
np.save('validation_c_index.npy',val_c_index_total)
np.save('train_c_index.npy',val_c_index_total)

print("Training complete!")
