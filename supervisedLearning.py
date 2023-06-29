import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataLoader import TradeDataset
from ForexNet import ForexNN


def saveModel(model, filePath):
    torch.save(model.state_dict(), filePath)


def loadTrainModel(model, filePath):
    return model.load_state_dict(torch.load(filePath)).train()


def loadEvalModel(model, filePath):
    return model.load_state_dict(torch.load(filePath)).eval()


def saveCheckpoint(model, optimizer, epoch, filePath):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }
    torch.save(checkpoint, filePath)


def loadTrainCheckpoint(model, optimizer, filePath):
    checkpoint = torch.load(filePath)
    model.load_state_dict(checkpoint['model_state']).train()
    optimizer.load_state_dict(checkpoint['optim_state'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def loadEvalCheckpoint(model, optimizer, filePath):
    checkpoint = torch.load(filePath)
    model.load_state_dict(checkpoint['model_state']).eval()
    optimizer.load_state_dict(checkpoint['optim_state'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def testModel(model, testLoader, device):
    model.eval()
    correct = 0
    samples = 0
    for data, labels in testLoader:
        data = F.normalize(data)
        data = data.to(device)
        labels = torch.flatten(labels).to(device)
        outputs = model(data)

        _, predicted = torch.max(outputs.data, 1)
        samples += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = 100.0 * correct / samples
    print(f'Accuracy of the network on test data: {acc} %')
    print(correct)
    print(samples)


def main(inputSize, hiddenSize, outputSize, numEpochs, batchSize, learningRate):
    # Device Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data File Paths
    trainPath = './labelledTrain2.csv'
    testPath = './labelledTest.csv'

    # Data Loading
    trainData = TradeDataset(trainPath)
    testData = TradeDataset(trainPath)

    # Data Loader
    trainLoader = DataLoader(dataset=trainData,
                             batch_size=batchSize,
                             shuffle=True)
    testLoader = DataLoader(dataset=testData,
                            batch_size=batchSize,
                            shuffle=False)

    # setup Training requirements
    model = ForexNN(inputSize, hiddenSize, outputSize).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    # Train the model
    totalSteps = len(trainLoader)
    for epoch in range(numEpochs):
        for i, (data, labels) in enumerate(trainLoader):
            # Bring Tensors to Cuda
            data = F.normalize(data)
            data = data.to(device)
            labels = torch.flatten(labels).type(torch.LongTensor).to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(outputs, labels)
                print(
                    f'Epoch [{epoch + 1}/{numEpochs}], Step [{i + 1}/{totalSteps}], Loss: {loss.item():.4f}')

        if epoch % 50000 == 0:
            name = f'./models/forex_model_{epoch}.pth'
            saveModel(model, name)

    # test the model
    testModel(model, testLoader, device)
    name = './models/forex_model_final.pth'
    saveModel(model, name)


if __name__ == "__main__":
    # Hyper-parameters
    inputSize = 2400
    hiddenSize = 2000
    outputSize = 3
    numEpochs = 5000000
    batchSize = 100
    learningRate = 0.003

    main(inputSize, hiddenSize, outputSize, numEpochs, batchSize, learningRate)
