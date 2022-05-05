import numpy as np
from torchvision.transforms import transforms

from dataset import CustomDataset
import torch

from torch.utils.data.dataset import Dataset  # For custom datasets
import matplotlib.pyplot as plt
import torch.optim as optim
from scipy.ndimage import rotate

from dense_net import DenseNet
from model import CTNetModel
from paper_model import PaperModel


def perf_measure(y_actual, y_hat, is_test, print_values=False):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    if not is_test:
        for i in range(len(y_actual)):
            if print_values:
                print(type(y_hat[i]))
                print(type(y_actual[i]))

            if y_actual[i] == y_hat[i] == 1:
                TP += 1
            if y_hat[i] == 1 and (y_actual[i] != y_hat[i]):
                FP += 1
            if y_actual[i] == y_hat[i] == 0:
                TN += 1
            if y_hat[i] == 0 and (y_actual[i] != y_hat[i]):
                FN += 1
    if is_test:
        if y_actual == y_hat == 1:
            TP += 1
        if y_hat == 1 and (y_actual != y_hat):
            FP += 1
        if y_actual == y_hat == 0:
            TN += 1
        if y_hat == 0 and (y_actual != y_hat):
            FN += 1

    return TP, FP, TN, FN


transformations = transforms.Compose([
                                      transforms.RandomRotation(15),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomHorizontalFlip()])
train_dataset = CustomDataset('labels_train_nobed.txt', augmentation=True)
val_dataset = CustomDataset('labels_val_nobed.txt')
# test_dataset = CustomDataset('test_labels.txt')


dataloader_train = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32,
                                               shuffle=True)

dataloader_val = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=32,
                                             shuffle=True)

# dataloader_test = torch.utils.data.DataLoader(dataset=test_dataset)
print(len(dataloader_train))
torch.manual_seed(5255)

model = CTNetModel()
binary_loss = torch.nn.BCELoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
optimizer = torch.optim.SGD(model.parameters(), 0.0001, momentum=0.5)


train_losses = []
validation_losses = []
max_value = 0.0
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)
train_acc = []
val_acc = []
for e in range(50):
    tp, fp, tn, fn = 0, 0, 0, 0
    model.train()
    train_loss = []
    correct = 0
    # if (e % 30) == 0:
    #     for g in optimizer.param_groups:
    #         g['lr'] = 0.00005

    for i, batch in enumerate(dataloader_train):
        x, y = batch
        optimizer.zero_grad()
        out = model(x)

        temp_max_value = torch.max(out)
        if temp_max_value > max_value:
            max_value = temp_max_value
        loss = binary_loss(torch.squeeze(out), torch.tensor(np.array(y).astype(np.float32)))
        train_loss.append(loss.item())
        # Replaces pow(2.0) with abs() for L1 regularization
        acc = torch.sum(torch.where(torch.squeeze(out) >= 0.5, 1, 0) == torch.tensor(np.array(y).astype(np.float32)))
        correct += acc.item()

        loss.backward()
        optimizer.step()
        # if i % 10 == 0:
        #     print("Train loss at epoch: {} step {}: {}".format(e, i, loss.item()))
    print("Max prediction for this epoch was {}".format(max_value))
    print("Train loss at epoch {}".format(np.mean(train_loss)))
    print("Train acc at epoch {}: {}".format(e, correct / len(train_dataset)))
    train_losses.append(np.mean(train_loss))
    train_acc.append(correct/len(train_dataset))
    model.eval()
    # print(scheduler.get_lr())
    scheduler.step()
    with torch.no_grad():
        val_losses = []
        val_sensitivities = []
        correct = 0
        for i, batch in enumerate(dataloader_val):
            x, y = batch
            out = model(x)
            loss = binary_loss(torch.squeeze(out), torch.tensor(np.array(y).astype(np.float32)))
            acc = torch.sum(torch.where(torch.squeeze(out) >= 0.5, 1, 0) == torch.tensor(np.array(y).astype(np.float32)))
            batch_tp, batch_fp, batch_tn, batch_fn = perf_measure(np.array(y).astype(np.float32), torch.where(torch.squeeze(out) >= 0.5, 1, 0), False, print_values=False)
            tp += batch_tp
            fp += batch_fp
            tn += batch_tn
            fn += batch_fn
            correct += acc.item()
            val_losses.append(loss.item())

        val_acc.append(correct/len(val_dataset))
        val_sensitivity = tp / (tp + fn)
        print("Val loss at epoch {}: {}".format(e, np.mean(val_losses)))
        print("Val acc at epoch {}: {}".format(e, correct / len(val_dataset)))
        print("Val sensitivity at epoch {}: {}".format(e, val_sensitivity))
        print("TP: {}, FP: {}, TN: {}, FN: {}".format(tp, fp, tn, fn))
    validation_losses.append(np.mean(val_losses))
    torch.save(model.state_dict(), 'model-simple.pth')

# plot losses
plt.plot(train_losses, '-o')
plt.plot(validation_losses, '-o')
plt.xlabel('epoch')
plt.ylabel('losses')
plt.legend(['Train', 'Valid'])
plt.title('Train vs Valid Losses')

# plot accuracy
plt.figure()
plt.plot(train_acc, '-o')
plt.plot(val_acc, '-o')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train', 'Valid'])
plt.title('Train vs Valid Accuracy')

plt.show()

plt.show()
#test on unseen data
