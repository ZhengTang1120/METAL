from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# hyperparameters
lr = 1e-3
weight_decay = 1e-5
batch_size = 100
shuffle = True
n_epochs = 10
vectors = glove.vectors
hidden_size = 100
num_layers = 2
bidirectional = True
dropout = 0.1
output_size = len(index_to_tag)

# initialize the model, loss function, optimizer, and data-loader
model = MyModel(vectors, hidden_size, num_layers, bidirectional, dropout, output_size).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
train_ds = MyDataset(train_df['word ids'], train_df['tag ids'])
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
dev_ds = MyDataset(dev_df['word ids'], dev_df['tag ids'])
dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

train_loss, train_acc = [], []
dev_loss, dev_acc = [], []