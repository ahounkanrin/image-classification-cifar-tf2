import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm
import os
from models.alexnet import AlexNet
from models.vgg import VGG16

parser = argparse.ArgumentParser()
# parser.add_argument("--epochs", default=100, type=int, help="Number of trainig epochs")
parser.add_argument("--lr", default=1e-2, type=float, help="Initial learning rate")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--dataset", required=True, type=str, 
                    help="Dataset on which the model should be trained")
args = parser.parse_args()

DATA_DIR = "./"

class ModelTrainer:
    def __init__(self) -> None:
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
        self.train_loss = tf.keras.metrics.CategoricalCrossentropy(name="loss")
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        self.test_loss = tf.keras.metrics.CategoricalCrossentropy(name="test_loss")
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
        self.model = AlexNet(num_classes=10)
        # self.model = VGG16(num_classes=10)

    def preprocess_training_data(self, x, y):
        # data_augmentation = tf.keras.Sequential([
        #                     tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        #                     tf.keras.layers.RandomRotation(0.2),
        #                     tf.keras.layers.Rescaling(1./255)
        #                     ])
        # x = data_augmentation(x)
        x = tf.image.flip_left_right(x)
        x = tf.image.flip_up_down(x)
        x = tf.cast(x, dtype=tf.float32)/255.
        if args.dataset.lower() == "cifar10" or args.dataset.lower() == "mnist":
            nclasses = 10
        elif args.dataset.lower() == "cifar100":
            nclasses = 100
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported.")
        y = tf.one_hot(y, depth=nclasses)
        return x, tf.squeeze(y)
    
    def preprocess_test_data(self, x, y):
        # data_augmentation = tf.keras.Sequential([
        #                     tf.keras.layers.Rescaling(1./255)
        #                     ])
        # x = data_augmentation(x)
        x = tf.cast(x, dtype=tf.float32)/255.
        if args.dataset.lower() == "cifar10" or args.dataset.lower() == "mnist":
            nclasses = 10
        elif args.dataset.lower() == "cifar100":
            nclasses = 100
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported.")
        y = tf.one_hot(y, depth=nclasses)
        return x, tf.squeeze(y)
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_object(y, y_pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss.update_state(y, y_pred)
        self.train_accuracy.update_state(y, y_pred)

    @tf.function
    def test_step(self, x, y):
        y_pred = self.model(x)
        self.test_accuracy.update_state(y, y_pred)
        self.test_loss.update_state(y, y_pred)
 

    def main(self):
        # self.model.build(input_shape=(None, 224, 224, 3))
        # self.model.summary()
        if args.dataset.lower() == "cifar10":
            _, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        elif args.dataset.lower() == "cifar100":
            _, (x_test, y_test)= tf.keras.datasets.cifar100.load_data()  
        elif args.dataset.lower() == "mnist":
            _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported.")     

        data_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)

        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        checkpoint_dir = os.path.join(DATA_DIR, "checkpoints")
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)
        checkpoint.restore(manager.checkpoints[-1]).expect_partial()

        for test_images, test_labels in tqdm(data_test.map(map_func=self.preprocess_test_data, 
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)):
            self.test_step(test_images, test_labels)

        print(f" test_loss = {self.test_loss.result():.6f}\ttest_acc = {self.test_accuracy.result():.6f}\
               \t model restored from {manager.latest_checkpoint}")
        
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.main()

        




