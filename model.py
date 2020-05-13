import tensorflow as tf

def Model_3():
  model = tf.keras.Sequential()

  model.add(tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same',
                                   input_shape=(96,96,1)
                                   ))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same'
                                   ))
  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same'
                                   ))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same'
                                   ))
  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(filters=96,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same'
                                   ))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(filters=96,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same'
                                   ))
  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same'
                                   ))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same'
                                   ))
  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same'
                                   ))
  model.add(tf.keras.layers.BatchNormalization())
  
  model.add(tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same'
                                   ))
  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Conv2D(filters=512,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same'
                                   ))
  model.add(tf.keras.layers.BatchNormalization())
  
  model.add(tf.keras.layers.Conv2D(filters=512,
                                   kernel_size=(3,3),
                                   activation='relu',
                                   padding='same'
                                   ))
  model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
  model.add(tf.keras.layers.BatchNormalization())

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(512, activation='relu'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dropout(0.1))
  model.add(tf.keras.layers.Dense(30))

  return model
