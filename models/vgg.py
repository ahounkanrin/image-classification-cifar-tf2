import tensorflow as tf

SEED = 1729

class VGG16(tf.keras.Model):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1_1 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding="same", activation="relu")
        self.conv1_2 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding="same", activation="relu")
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv2_1 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu")
        self.conv2_2 = tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu")
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)
        self.conv3_1 = tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding="same", activation="relu")
        self.conv3_2 = tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding="same", activation="relu")
        self.conv3_3 =  tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding="same", activation="relu")
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.conv4_1 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding="same", activation="relu")
        self.conv4_2 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding="same", activation="relu")
        self.conv4_3 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding="same", activation="relu")
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.conv5_1 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding="same", activation="relu")
        self.conv5_2 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding="same", activation="relu")
        self.conv5_3 = tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding="same", activation="relu")
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.5, seed=SEED) 
        self.fc2 = tf.keras.layers.Dense(4096, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.5, seed=SEED)
        self.fc3 = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout1(x)
        x = self.fc2(x)
        if training:
            x = self.dropout2(x)
        outputs = self.fc3(x)
        return outputs
    
if __name__ == "__main__":
    tf.keras.utils.set_random_seed(seed=SEED)
    tf.config.experimental.enable_op_determinism()

    model = VGG16(num_classes=1000)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

    # inputs = tf.random.uniform(shape=(32, 224, 224, 3), seed=SEED)
    # predictions = model(inputs, training=True)
    # model.summary()