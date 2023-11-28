import tensorflow as tf
# from tensorflow.keras import layers 

SEED = 1729

class AlexNet(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(96, kernel_size=(11,11), padding="same", activation="relu")
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=(5,5), padding="same", activation="relu")
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
        self.conv3 = tf.keras.layers.Conv2D(384, kernel_size=(3,3), padding="same", activation="relu")
        self.conv4 = tf.keras.layers.Conv2D(384, kernel_size=(3,3), padding="same", activation="relu")
        self.conv5 = tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding="same", activation="relu")
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(4096, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.5, seed=SEED)
        self.dense2 = tf.keras.layers.Dense(4096, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.5, seed=SEED)
        self.dense3 = tf.keras.layers.Dense(num_classes, activation="softmax")


    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout1(x)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x)
        outputs = self.dense3(x)
        return outputs

    


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    model = AlexNet(num_classes=10)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

    # inputs = tf.random.uniform(shape=(2, 224, 224, 3), seed=SEED)
    # predictions=  model(inputs, training=True)
    # model.summary()