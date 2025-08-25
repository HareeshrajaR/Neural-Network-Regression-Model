# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.
## Neural Network Model

![image](https://github.com/user-attachments/assets/84093ee0-48a5-4bd2-b78d-5d8ee258d189)


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
### Name: HAREESH R
### Register Number:212223230068
```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.history={'loss':[]}

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


hari_brain=NeuralNetwork()
criteria=nn.MSELoss()
optimizer=optim.RMSprop(hari_brain.parameters(), lr=0.001)



def train_model(hari_brain,X_train,y_train,criteria,optimizer,epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      output = hari_brain(X_train)
      loss = criteria(output, y_train)
      loss.backward()
      optimizer.step()

      hari_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

      
    



```
## Dataset Information

<img width="352" height="758" alt="image" src="https://github.com/user-attachments/assets/96825e48-7668-4145-aa4f-adaf1098d0f9" />


## OUTPUT

### Training Loss Vs Iteration Plot
<img width="796" height="519" alt="image" src="https://github.com/user-attachments/assets/11438f4d-df80-44a1-adc3-9b2e82aafedc" />




### New Sample Data Prediction
<img width="924" height="137" alt="image" src="https://github.com/user-attachments/assets/ffe2180f-8c5c-430e-97c9-2c1ece3ec23b" />




## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
