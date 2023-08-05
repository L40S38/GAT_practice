import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


# データの前処理
def load_cora_data():
    # この部分はCoraデータセットの読み込みと前処理を行う関数に置き換えてください
    # 以下はダミーデータを使用する例です
    num_nodes = 2708
    num_features = 1433
    num_classes = 7

    # ダミーデータの生成
    features = np.random.randn(num_nodes, num_features)
    labels = np.random.randint(num_classes, size=num_nodes)
    #adj = sp.random(num_nodes, num_nodes, density=0.1, format="coo") # ダミーグラフの隣接行列
    adj = np.random.randint(num_nodes, size=(num_nodes,num_nodes))

    features = torch.from_numpy(features).float()
    #labels = torch.from_numpy(labels).float()
    #adj = torch.from_numpy(adj).float()

    return features, labels, adj

# 正規化した隣接行列を取得
def normalize_adj(adj):
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return torch.from_numpy(np.array(adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo())).float()

# Graph Attention Layer
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        print(f"adj:{adj.shape},e:{e.shape},zero_vec:{zero_vec.shape}")
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)
        return h_prime

# Graph Attention Network
class GAT(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, n_heads, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.dropout = dropout
        self.alpha = alpha

        self.attentions = nn.ModuleList([GraphAttentionLayer(n_features, n_hidden, dropout=dropout, alpha=alpha) for _ in range(n_heads)])
        self.out_att = GraphAttentionLayer(n_hidden * n_heads, n_classes, dropout=dropout, alpha=alpha)

    def forward(self, input, adj):
        x = input
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

# データセットのクラス
class CoraDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# モデルの学習
def train_model(model, optimizer, criterion, data_loader, adj):
    model.train()
    total_loss = 0.0
    for features, labels in data_loader:
        optimizer.zero_grad()
        output = model(features, adj)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
    return total_loss / len(data_loader.dataset)

# モデルの評価
def evaluate_model(model, criterion, data_loader, adj):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for features, labels in data_loader:
            output = model(features, adj)
            total_loss += criterion(output, labels).item() * features.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    return total_loss / len(data_loader.dataset), correct / len(data_loader.dataset)

def main():
    # データセットのロードと前処理
    features, labels, adj = load_cora_data()
    n_features = features.shape[1]
    n_classes = len(np.unique(labels))
    n_nodes = features.shape[0]
    print(f"features:{features.shape},labels:{labels.shape},adj:{adj.shape}")
    print(f"features:{features.dtype},labels:{labels.dtype},adj:{adj.dtype}")

    # 正規化した隣接行列を取得
    adj_normalized = normalize_adj(adj)
    print(f"adj_normalized:{adj_normalized.shape}")

    # データセットの分割
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, stratify=labels)
    train_dataset = CoraDataset(train_features, train_labels)
    test_dataset = CoraDataset(test_features, test_labels)

    # DataLoaderの作成
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # モデルの定義
    n_hidden = 8
    n_heads = 8
    model = GAT(n_features, n_classes, n_hidden, n_heads)

    # 損失関数と最適化手法の定義
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # モデルの学習と評価
    n_epochs = 100
    for epoch in range(n_epochs):
        train_loss = train_model(model, optimizer, criterion, train_loader, adj_normalized)
        test_loss, test_accuracy = evaluate_model(model, criterion, test_loader, adj_normalized)
        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
