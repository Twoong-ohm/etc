# -*- coding: utf-8 -*-
"""keras.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/tensorflow/docs/blob/master/site/ko/tutorials/distribute/keras.ipynb

##### Copyright 2019 The TensorFlow Authors.
"""


from __future__ import absolute_import, division, print_function, unicode_literals

# 텐서플로와 텐서플로 데이터셋 패키지 가져오기
!pip install tensorflow-gpu==2.0.0-rc1
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

import os


datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']

strategy = tf.distribute.MirroredStrategy()

print('장치의 수: {}'.format(strategy.num_replicas_in_sync))


num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label


train_dataset = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)


with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])


# 체크포인트를 저장할 체크포인트 디렉터리를 지정합니다.
checkpoint_dir = './training_checkpoints'
# 체크포인트 파일의 이름
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# 학습률을 점점 줄이기 위한 함수
# 필요한 함수를 직접 정의하여 사용할 수 있습니다.
def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5

# 에포크가 끝날 때마다 학습률을 출력하는 콜백.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\n에포크 {}의 학습률은 {}입니다.'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]


model.fit(train_dataset, epochs=12, callbacks=callbacks)



model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_dataset)

print('평가 손실: {}, 평가 정확도: {}'.format(eval_loss, eval_acc))


path = 'saved_model/'

tf.keras.experimental.export_saved_model(model, path)


unreplicated_model = tf.keras.experimental.load_from_saved_model(path)

unreplicated_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy'])

eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)

print('평가 손실: {}, 평가 정확도: {}'.format(eval_loss, eval_acc))

with strategy.scope():
  replicated_model = tf.keras.experimental.load_from_saved_model(path)
  replicated_model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])

  eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
  print ('평가 손실: {}, 평가 정확도: {}'.format(eval_loss, eval_acc))