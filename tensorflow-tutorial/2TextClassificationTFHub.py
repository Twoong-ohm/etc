import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow import keras

train_validation_split = tfds.Split.TRAIN.subsplit([6,4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True
)

embedding = ["https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
             "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
             "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"]

hub_layer = hub.KerasLayer(embedding[0], input_shape=[],
                           dtype=tf.string,
                           trainable=True,)

model = keras.Sequential()
model.add(hub_layer)
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))


