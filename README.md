# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Design and implement a Convolutional Neural Network (CNN) to classify grayscale images from the FashionMNIST dataset into 10 distinct categories. The model should learn to recognize patterns and features in the images to accurately predict their respective classes.

![{7C35D827-9878-4F05-99AA-CBE168EB7B6F}](https://github.com/user-attachments/assets/9382e182-1012-458b-92c0-968bd1cd6710)


## Neural Network Model

![{F2C9AC2C-267A-417E-8265-BF7534819113}](https://github.com/user-attachments/assets/b876b3eb-421a-44c7-996b-12e969981eac)


## DESIGN STEPS

### STEP 1: 
Classify grayscale images into 10 categories using a CNN.

### STEP 2: 
Load the FashionMNIST dataset with 60,000 training and 10,000 test images.

### STEP 3: 
Convert images to tensors, normalize, and create DataLoaders for efficient processing.

### STEP 4:
Build a CNN with convolution, activation, pooling, and fully connected layers.

### STEP 5: 
Train the model using CrossEntropyLoss and Adam optimizer over multiple epochs.

### STEP 6: 
Test the model, compute accuracy, and analyze results using a confusion matrix and classification report.

### STEP 7:
Predict new images and display actual vs. predicted labels for visual analysis.


## PROGRAM

### Name:SREE NIVEDITAA SARAVANAN
### Register Number:
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x): 
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

![{5093A2EE-4509-41A1-81C0-994F0AC31FA9}](https://github.com/user-attachments/assets/590c5f79-0b9c-4128-b105-887660f9a413)


### Confusion Matrix

![{41E349A3-51D8-4D65-833E-16350CE5618F}](https://github.com/user-attachments/assets/519374ec-51df-4a4c-b21d-afe001ae2fdc)


### Classification Report

![{A4C954A7-907B-4668-BB09-B99DB47376D8}](https://github.com/user-attachments/assets/cc09e5f3-00d7-48af-9668-6bc683c9a2d1)



### New Sample Data Prediction

![{EE0D1A2A-FFB9-49C0-9A06-ACA2BD73A406}](https://github.com/user-attachments/assets/4c4da847-c19d-47e0-b115-e62182a70a91)

## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.

