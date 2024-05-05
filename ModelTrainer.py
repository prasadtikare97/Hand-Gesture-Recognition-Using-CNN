import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Label encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(data_dict['labels'])

# Pad data sequences and normalize
max_length = max(len(seq) for seq in data_dict['data'])
data_padded = [seq + [0] * (max_length - len(seq)) for seq in data_dict['data']]
data = np.array(data_padded, dtype='float32')
data /= np.max(data)  # Normalization

# Convert labels to categorical
labels = keras.utils.to_categorical(encoded_labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, shuffle=True, random_state=42)

# Reshape input data
x_train = x_train.reshape(-1, max_length, 1)
x_test = x_test.reshape(-1, max_length, 1)

# Build the CNN model
model = keras.Sequential([
    layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(max_length, 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.25),
    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(np.unique(encoded_labels)), activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_labels, y_pred_labels)
print('{}% of samples were classified correctly!'.format(accuracy * 100))

# Save the model
model.save('model.h5')
