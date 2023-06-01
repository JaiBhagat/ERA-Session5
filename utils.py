import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []
test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def preprocess_data(data):
    """
    Preprocesses the MNIST data based on the specified data type.

    Args:
        data (str): Type of data to preprocess. Should be either 'train' or 'test'.

    Returns:
        transforms.Compose: Preprocessed data transformations.
    """
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22)], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    if data == 'train':
        return train_transforms
    elif data == 'test':
        return test_transforms
    else:
        raise ValueError("Invalid argument. Use 'train' or 'test'.")

def plot_data(load_data, img_cnt):
    """
    Plots sample images and their labels from a data loader.

    Args:
        load_data (DataLoader): Data loader to extract images and labels from.
        img_cnt (int): Number of images to plot.
    """
    batch_data, batch_label = next(iter(load_data))

    fig = plt.figure()

    for i in range(img_cnt):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def get_correct_pred_count(prediction, labels):
    """
    Calculates the number of correct predictions.

    Args:
        prediction (Tensor): Predicted labels.
        labels (Tensor): Ground truth labels.

    Returns:
        int: Number of correct predictions.
    """
    return prediction.argmax(dim=1).eq(labels).sum().item()

def train(model, device, train_loader, optimizer, train_acc):
    """
    Trains the model on the training dataset.

    Args:
        model (nn.Module): Model to train.
        device (torch.device): Device to run the training on.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (optim.Optimizer): Optimizer for model parameter updates.

    Returns:
        float: Training loss.
        float: Training accuracy.
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0
    train_acc = []  # Initialize train_acc list

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = nn.functional.nll_loss(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += get_correct_pred_count(pred, target)
        processed += len(data)

        pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)  # Append to train_acc list
    train_losses.append(train_loss/len(train_loader))

    return train_loss/len(train_loader), train_acc[-1]


def test(model, device, test_loader,test_acc):
    """
    Evaluates the model on the test dataset.

    Args:
        model (nn.Module): Model to evaluate.
        device (torch.device): Device to run the evaluation on.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        float: Test loss.
        float: Test accuracy.
    """
    model.eval()

    test_loss = 0
    correct = 0
    total_samples = 0
    test_acc = []  # Initialize test_acc list

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()

            # Get predictions and calculate accuracy
            _, predicted = torch.max(output, dim=1)
            correct += (predicted == target).sum().item()
            total_samples += target.size(0)

    test_loss /= total_samples
    test_accuracy = 100.0 * correct / total_samples
    test_acc.append(test_accuracy)  # Append to test_acc list
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, total_samples, test_accuracy))

    return test_loss, test_accuracy

