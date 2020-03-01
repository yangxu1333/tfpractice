import tensorflow as tf
from tfrloader import train_ds
units = 64
classes = 101 #UCF-101
epochs = 5

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units, return_sequences=True),
    tf.keras.layers.Dense(classes, activation='relu'),
])
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_ds, epochs=epochs)
""" for epoch in range(epochs):
    for x in train_ds:
        model.fit(x[0],x[1]) """
