import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt

#### requiring same set of train and validation data for all the run
np.random.seed(1223)


#### reading the csv file
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#### managing dataset
df_features = df_train.iloc[:, 1:785]
df_labels = df_train.iloc[:, 0]
x_test = df_test.iloc[:, 0:784]

#### spliting dataset to train and test
x_train, x_validation, y_train, y_validation = train_test_split(df_features, df_labels, test_size=0.2, random_state=1223)
x_train = x_train.as_matrix().reshape(33600,784)
x_validation = x_validation.as_matrix().reshape(8400, 784)
x_test = x_test.as_matrix().reshape(28000, 784)


#### data normalization
x_train = x_train.astype('float32') / 255
x_validation = x_validation.astype('float32') / 255
x_test = x_test.astype('float32') / 255


#### encoding lebels
y_train = to_categorical(y_train)
y_validation = to_categorical(y_validation)


#### creating the model
#model = models.Sequential()
#model.add(layers.Dense(512, activation='relu', input_shape=(784,)))
#model.add(layers.Dense(10, activation='softmax'))







def create_model(units_hidden_layers, input_shape):
    model = models.Sequential()
    model.add(layers.Dense(units_hidden_layers[0], activation='relu', input_shape=(input_shape,)))
    hidden_layers = len(units_hidden_layers)
    for i in range(1, hidden_layers - 1):
        model.add(layers.Dense(units_hidden_layers[i], activation='relu'))

    model.add(layers.Dense(units_hidden_layers[hidden_layers - 1], activation='softmax'))
    return model


#### generating model using create_model() function
units_in_h_l = [512, 256, 128, 64, 10]
input_shape = 784
model = create_model(units_in_h_l, input_shape)


#print(model.summary())


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=200, batch_size=256, validation_data=(x_validation, y_validation))
print(history.history['acc'])


test_pred = pd.DataFrame(model.predict(x_test, batch_size=256))
test_pred = pd.DataFrame(test_pred.idxmax(axis=1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1

test_pred.to_csv('mnist_submission.csv', index = False)
