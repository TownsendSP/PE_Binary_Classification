from dataLoader_Wrapper import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
# test the model and generate ROC curve and conf mat
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, random_split

from BinaryClassifier import *
from dataLoader_Wrapper import *
import tqdm


def main():
    dataThing = da.DataAcquirer('H:/Datasets/Bigger_dataset/data.sqlite', 'H:/Datasets/Bigger_dataset/dataset/')
    dataThing.bootup()
    num_chunks = 5000
    chunk_length = 256
    x86rows = dataThing.filter('binaries', 'platform', 'x86')
    x64rows = dataThing.filter('binaries', 'platform', 'x64')


    print("Getting file info")
    x86file_info = dataThing.getFileInfo(x86rows)
    x64file_info = dataThing.getFileInfo(x64rows)
    print("Got file info")

    x86chunkList = extractChunks(x86file_info, chunk_length, num_chunks)
    print(f"length of x86chunkList: {len(x86chunkList)}")
    # print(f"size of x86chunkList[0]: {len(x86chunkList[0])}")
    # [print(f"x86[{i}]: {len(x86chunkList[i])}") for i in range(len(x86chunkList))]
    x64chunkList = extractChunks(x64file_info, chunk_length, num_chunks)
    # [print(f"x64[{i}]: {len(x64chunkList[i])}") for i in range(len(x64chunkList))]

    # remove entries from the larger list until they are the same size
    if len(x86chunkList) > len(x64chunkList):
        x86chunkList = x86chunkList[:len(x64chunkList)]
    elif len(x64chunkList) > len(x86chunkList):
        x64chunkList = x64chunkList[:len(x86chunkList)]

    print("Extracted chunks")
    print(f'x86 chunks: {len(x86chunkList)}')
    print(f'x64 chunks: {len(x64chunkList)}')


    dataset = BinaryClassificationDataset(x86chunkList, x64chunkList)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])



    chunks = x64chunkList
    print(f"length of chunks: {len(chunks)}")
    print(f"length of chunks[-1]: {len(chunks[0])}")
    print(f"datatype of first element of chunks: {type(chunks[0])}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")
    model = BinaryClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.011)
    criterion = nn.BCELoss()


    # initialize the data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # print the number of batches in each loader
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")


    # train the model
    num_epochs = 100

    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            # x = x.to(device)
            x = x.view(-1, chunk_length, 1).to(device)  # reshape x
            # y = y.to(device)
            y = y.view(-1, 1).float().to(device)  # reshape y
            # forward pass
            scores = model(x)
            loss = criterion(scores, y)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # gradient descent or adam step
            optimizer.step()
            # compute accuracy on validation set
            correct = 0
            total = 0
        with torch.no_grad():
            for x_val, y_val in train_loader:
                x_val = x_val.view(-1, chunk_length, 1).to(device)
                y_val = y_val.view(-1, 1).float().to(device)
                scores = model(x_val)
                predictions = (scores > 0.5).float()
                correct += (predictions == y_val).sum().item()
                total += y_val.size(0)
        accuracy = correct / total

        print(f"Epoch {epoch} | Loss: {loss.item()} | Accuracy: {accuracy}")
    # best_accuracy = 0.0
    # for epoch in range(num_epochs):
    #     for i, (x, y) in enumerate(train_loader):
    #         x = x.view(-1, chunk_length, 1).to(device)  # reshape x
    #         y = y.view(-1, 1).float().to(device)  # reshape y
    #         scores = model(x)
    #         loss = criterion(scores, y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     # compute accuracy on validation set
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for x_val, y_val in test_loader:
    #             x_val = x_val.view(-1, chunk_length, 1).to(device)
    #             y_val = y_val.view(-1, 1).float().to(device)
    #             scores = model(x_val)
    #             predictions = (scores > 0.5).float()
    #             correct += (predictions == y_val).sum().item()
    #             total += y_val.size(0)
    #     accuracy = correct / total
    #
    #     # update best accuracy and save model if it's the best so far
    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         torch.save(model.state_dict(), 'best_model.pth')
    #
    #     print(f"Epoch {epoch} | Accuracy: {accuracy}")

    # save the model
    torch.save(model.state_dict(), "model.ckpt")
    print(f"Model saved to model.ckpt")


    model = BinaryClassifier().to(device)
    model.load_state_dict(torch.load("model.ckpt"))

    # use dataGetter

    # dataset = BinaryClassificationDataset(x86chunkList, x64chunkList)
    # test_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # test the model
    y_test = []
    y_pred = []
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_test.extend(y_batch.tolist())
        with torch.no_grad():
            scores = model(x_batch)
            preds = torch.round(scores)
            y_pred.extend(preds.tolist())



    # generate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)


    # plot the ROC curve
    plt.plot(fpr, tpr, label=f"AUC: {auc}")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


    # generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # make rows and columns for x86 and x64
    cm_df = pd.DataFrame(cm, index=["x86", "x64"], columns=["x86", "x64"])
    # plot the confusion matrix
    sns.heatmap(cm_df, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    main()