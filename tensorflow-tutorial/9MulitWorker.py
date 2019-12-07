import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

datasets, info = tfds.load(name='mnist', with_info = True, as_supervised=True)

train_datasets_unbatched = datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)
train_datasets = train_datasets_unbatched.batch(BATCH_SIZE)

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(
      loss=tf.keras.losses.sparse_categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

NUM_WORKERS = 2

GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)
with strategy.scope():
  multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(x=train_datasets, epochs=3)



