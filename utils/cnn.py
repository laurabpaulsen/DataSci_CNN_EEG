import numpy as np

# sklearn tools
from sklearn.model_selection import train_test_split

# pytorch tools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class GAFDataset(Dataset):
    """Dataset class for GAF images, inherits from torch.utils.data.Dataset"""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.labels[idx]
        return X, y
 
def prep_dataloaders(gafs, labels, batch_size=4, test_size=0.3):
    """
    Creates dataloaders for training, validation, and testing

    Parameters
    ----------
    gafs : np.array
        The gaf images
    labels : np.array
        array of labels
    batch_size : int, optional
        Batch size, by default 4

    Returns
    -------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    test_loader : DataLoader
        Testing data loader
    y_test : np.array
        Labels for the test set
    """

    # split into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(gafs, labels, test_size=test_size, random_state=7)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=7)

    # create dataloaders
    train_loader = DataLoader(GAFDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(GAFDataset(X_val, y_val), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(GAFDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, y_test


# define a class for the CNN
class CNN():
    # initialize the class
    def __init__(self, model, optimizer, criterion, lr = 0.001):
        self.lr = lr
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    # train the model for one epoch
    def train(self, train_loader:DataLoader):
        """
        Train the model for one epoch and return the loss and accuracy

        Parameters
        ----------
        train_loader : DataLoader
            The training data loader
        
        Returns
        -------
        train_loss : float
            The training loss
        train_acc : float
            The training accuracy
        """
        self.model.train()
        train_loss, train_acc = 0.0, 0.0
        
        for X, y in train_loader:
            self.optimizer.zero_grad()
            y_hat = self.model(X.float())
            loss = self.criterion(y_hat.view(-1), y.float())
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_acc += ((torch.round(torch.sigmoid(y_hat.view(-1)))==y).sum().item() / len(y))

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        return train_loss, train_acc
    
    # train the model for X epochs
    def train_model(self, train_loader:DataLoader, val_loader:DataLoader, epochs: int):
        """Train the model and return the losses and accuracies
        
        Parameters
        ----------
        train_loader : DataLoader
            The training data loader
        val_loader : DataLoader
            The validation data loader
        epochs : int
            The number of epochs to train for
        
        Returns
        -------
        history : dict
            Dictionary with the train and validation loss and accuracies. 
        """

        # dict for storing losses and accuracies
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        for epoch in range(epochs):
            # train
            train_loss, train_acc = self.train(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # validate
            val_loss, val_acc = self.validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")

        return history
    
    def validate(self, val_loader:DataLoader): 
        self.model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for X, y in val_loader:
                y_hat = self.model(X.float())
                loss = self.criterion(y_hat.view(-1), y.float())
                val_loss += loss.item()
                val_acc += ((torch.round(torch.sigmoid(y_hat.view(-1)))==y).sum().item() / len(y))

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

        return val_loss, val_acc
    

    def predict(self, test_loader):
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for X, y in test_loader:
                y_hat = self.model(X.float())
                y_pred.append(torch.sigmoid(y_hat).numpy())

        return np.concatenate(y_pred)
    
    def state_dict(self):
        return self.model.state_dict()


class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv3d(50, 18, kernel_size=3)
            self.pool1 = nn.MaxPool3d(kernel_size=1)
            self.bn1 = nn.BatchNorm3d(18)
            self.drop1 = nn.Dropout3d(p=0.2)

            self.conv2 = nn.Conv3d(18, 128, kernel_size=(3, 3, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))
            self.bn2 = nn.BatchNorm3d(128)
            self.drop2 = nn.Dropout3d(p=0.2)
            
            self.conv3 = nn.Conv3d(128, 128, kernel_size=(3, 3, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))
            self.bn3 = nn.BatchNorm3d(128)
            self.drop3 = nn.Dropout3d(p=0.2)

            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 2)

        def forward(self, x):
            # convert tensor to float to avoid error 
            x = x.float()
            x = self.drop1(self.bn1(self.pool1(torch.relu(self.conv1(x)))))
            x = self.drop2(self.bn2(self.pool2(torch.relu(self.conv2(x)))))
            x = self.drop3(self.bn3(self.pool3(torch.relu(self.conv3(x)))))
            x = self.avgpool(x)
            x = x.view(-1, 128)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x