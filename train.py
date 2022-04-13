import torch
import os
import torch.nn.functional as F

from model import InvoiceGCN

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

best_loss = 1e+6
epochs = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torch.load(os.path.join("./GCN_data/processed", 'train_data.dataset'))
test_data = torch.load(os.path.join("./GCN_data/processed", 'test_data.dataset'))
train_data = train_data.to(device)
test_data = test_data.to(device)

model = InvoiceGCN(input_dim=train_data.x.shape[1], chebnet=True)
model = model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.001, weight_decay=0.9
)

# class weights for imbalanced data
_class_weights = compute_class_weight(
    "balanced", train_data.y.unique().cpu().numpy(), train_data.y.cpu().numpy()
)

for epoch in range(1, epochs+1):
    model.train()
    optimizer.zero_grad()

    # NOTE: just use boolean indexing to filter out test data, and backward after that!
    # the same holds true with test data :D
    # https://github.com/rusty1s/pytorch_geometric/issues/1928
    loss = F.nll_loss(
        model(train_data), train_data.y - 1, weight=torch.FloatTensor(_class_weights).to(device)
    )
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        if epoch % 50 == 0:
            model.eval()

            # forward model
            for index, name in enumerate(['train', 'test']):
                _data = eval("{}_data".format(name))
                y_pred = model(_data).max(dim=1)[1]
                y_true = (_data.y - 1)
                acc = y_pred.eq(y_true).sum().item() / y_pred.shape[0]

                y_pred = y_pred.cpu().numpy()
                y_true = y_true.cpu().numpy()
                print("\t{} acc: {}".format(name, acc))
                # confusion matrix
                if name == 'test':
                    cm = confusion_matrix(y_true, y_pred)
                    class_accs = cm.diagonal() / cm.sum(axis=1)
                    print(classification_report(y_true, y_pred))

            loss_val = F.nll_loss(model(test_data), test_data.y - 1
            )
            fmt_log = "Epoch: {:03d}, train_loss:{:.4f}, val_loss:{:.4f}"
            print(fmt_log.format(epoch, loss, loss_val))
            print(">" * 50)
            if best_loss>loss_val:
                torch.save(model.state_dict(), f"./weights/best.pt")
