from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import numpy as np 

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


# model creation
model = Sequential()

# input and hidden layer 
model.add(Dense(units=4, input_dim=2, activation='relu'))


# output layer 
model.add(Dense(units=1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(X, y, epochs=5000, verbose=0)

# evaluate model
loss, accuracy = model.evaluate(X, y)
print(f'Model accuracy: {accuracy * 100:.2f}%')

# test model

predictions = model.predict(X)
print('Predictions:')
for i, prediction in enumerate(predictions):
    print(f'Input: {X[i]} -> Predicted: {prediction[0]:.4f}, Rounded: {round(prediction[0])}, Excpected: {y[i][0]}')
