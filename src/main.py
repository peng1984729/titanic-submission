import random, torch, matplotlib
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

seed = 214
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

df = pd.read_csv("data/train.csv")

y = df["Survived"].values
df = df.drop(columns = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"])
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"male" : 0, "female" : 1})
df["Embarked"] = df["Embarked"].map({"S" : 0, "C" : 1, "Q" : 2})
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
X = df.values.astype(np.float32)
y = y.astype(np.float32)
feature_cols = df.columns.tolist() 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = seed)
 
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
   
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
train_loader = DataLoader(MyDataset(X_train, y_train),
                          batch_size = 32, shuffle = True)
test_loader = DataLoader(MyDataset(X_test, y_test),
                          batch_size = 32, shuffle = False)

class MLP(nn.Module):
    def __init__(self, in_dim, hd = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hd),
            nn.ReLU(),
            nn.Linear(hd, hd // 2),
            nn.ReLU(),
            nn.Linear(hd // 2, 1)
        )
    def forward(self, x):
        return self.net(x)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            prob = torch.sigmoid(model(xb).squeeze(1))
            pred_label = (prob > 0.5).float()
            correct += (pred_label == yb).sum().item()
            total += yb.size(0)
    return correct / total

model = MLP(X_train.shape[1])

loss_fn = nn.BCEWithLogitsLoss()

opt = torch.optim.Adam(model.parameters(), lr = 1e-3)

epochs = 200
train_losses, train_accs, test_accs = [], [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb).squeeze(1)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    train_losses.append(avg_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    if epoch % 20 == 0:
        train_acc = evaluate(model, train_loader)
        test_acc = evaluate(model, test_loader)
        print(f"epoch {epoch:3d} | loss {total_loss/len(train_loader):.4f} "
                      f"| train acc {train_acc:.3f} | test acc {test_acc:.3f}")

plt.figure()
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("docs/loss_curve.png", dpi = 150)
plt.close()

plt.figure()
plt.plot(train_accs, label = "Train Accuracy")
plt.plot(test_accs, label = "Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy")
plt.legend()
plt.savefig("docs/accuracy_curve.png", dpi = 150)
plt.close()

print(f"\n最终测试集准确率: {test_accs[-1]:.4f}  ({test_accs[-1]*100:.2f}%)")

# 保存模型和预处理信息（题目要求：保存可重新加载的模型）
torch.save({
    "model_state": model.state_dict(),
    "mean": mean,
    "std": std,
    "feature_cols": feature_cols,
}, "src/titanic_model.pt")

def predict(pclass, sex, age, sibsp, parch, fare, embarked):
    sex_num = {"male": 0, "female": 1}[sex]
    embarked_num = {"S": 0, "C": 1, "Q": 2}[embarked]
    family_size = sibsp + parch + 1
    x = np.array([pclass, sex_num, age, sibsp, parch, fare, embarked_num, family_size],
                 dtype = np.float32)
    x = (x - mean) / std
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(x).unsqueeze(0)).squeeze(1)).item()
    survived = 1 if prob > 0.5 else 0
    return survived

print("预测:\n一次性输入7 个值，如3 male 22 1 0 7.25 S:")

pclass, sex, age, sibsp, parch, fare, embarked = input().split()
pclass, sibsp, parch = int(pclass), int(sibsp), int(parch)
age, fare = float(age), float(fare)

print("预测结果:", predict(pclass, sex, age, sibsp, parch, fare, embarked))