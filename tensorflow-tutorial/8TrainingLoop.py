import tensorflow as tf
import numpy as np
import os

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = test_images[...,None]
test_images = test_images[...,None]

train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

strategy = tf.distribute.MirroredStrategy()

BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10

with strategy.scope():
    train_dataset= tf.data.Dataset.from_tensor_slices((train_images,train_labels)).shuffle(BUFFER_SIZE)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

def create_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

  return model

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

with strategy.scope():
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      reduction=tf.keras.losses.Reduction.NONE)
  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

with strategy.scope():
  test_loss = tf.keras.metrics.Mean(name='test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')

with strategy.scope():
  model = create_model()

  optimizer = tf.keras.optimizers.Adam()

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

with strategy.scope():
  def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
      predictions = model(images, training=True)
      loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss

  def test_step(inputs):
    images, labels = inputs

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)

with strategy.scope():

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step,
                                                          args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))


    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        for x in test_dist_dataset:
            distributed_test_step(x)

        if epoch % 2 == 0:
            checkpoint.save(checkpoint_prefix)

        template = ("에포크 {}, 손실: {}, 정확도: {}, 테스트 손실: {}, "
                    "테스트 정확도: {}")
        print(template.format(epoch + 1, train_loss,
                              train_accuracy.result() * 100, test_loss.result(),
                              test_accuracy.result() * 100))

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()