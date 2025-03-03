# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model


![Screenshot (397)](https://github.com/user-attachments/assets/f5405cec-5f3f-4f05-ae4e-d863bc61688a)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Rahini A
### Register Number: 212223230165
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here
        self.fc1=nn.Linear(1,4)
        self.fc2=nn.Linear(4,6)
        self.fc3=nn.Linear(6,8)
        self.fc4=nn.Linear(8,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.relu(self.fc3(x))
    x=self.fc4(x)
    return x




# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()
   # Append loss inside the loop
        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information

![Screenshot (419)](https://github.com/user-attachments/assets/de2e88ee-dba8-4227-a490-301c0ab935c2)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot (420)](https://github.com/user-attachments/assets/2b47e6cc-5346-4ef1-9aaf-799813e88e6a)


### New Sample Data Prediction


![Screenshot (421)](https://github.com/user-attachments/assets/a75247ef-afb4-4522-84be-6b34f3e5f316)

## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
