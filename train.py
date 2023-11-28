import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm
import os
from models.alexnet import AlexNet
from models.vgg import VGG16

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=100, type=int, help="Number of trainig epochs")
parser.add_argument("--lr", default=1e-1, type=float, help="Initial learning rate")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--dataset", required=True, type=str, 
                    help="Dataset on which the model should be trained")
args = parser.parse_args()

DATA_DIR = "./"

class ModelTrainer:
    def __init__(self) -> None:
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.train_loss = tf.keras.metrics.CategoricalCrossentropy(name="loss")
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        self.test_loss = tf.keras.metrics.CategoricalCrossentropy(name="test_loss")
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, decay_steps=1000, 
                                                                     decay_rate=0.95, staircase=True)
        # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(args.lr, decay_steps=1000)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
        if args.dataset.lower() == "cifar10" or args.dataset.lower() == "mnist":
            nclasses = 10
        elif args.dataset.lower() == "cifar100":
            nclasses = 100
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported.")
        
        self.model = AlexNet(num_classes=nclasses)
        # self.model = VGG16(num_classes=nclasses)

    def preprocess_training_data(self, x, y):
        # data_augmentation = tf.keras.Sequential([
        #                     tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        #                     tf.keras.layers.RandomRotation(0.2),
        #                     tf.keras.layers.Rescaling(1./255)
        #                     ])
        # x = data_augmentation(x)
        # x = tf.image.flip_left_right(x)
        # x = tf.image.flip_up_down(x)
        x = tf.cast(x, dtype=tf.float32)/255.
        if args.dataset.lower() == "cifar10" or args.dataset.lower() == "mnist":
            nclasses = 10
        elif args.dataset.lower() == "cifar100":
            nclasses = 100
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported.")
        y = tf.one_hot(y, depth=nclasses)
        return x, tf.squeeze(y)
    
    def preprocess_val_data(self, x, y):
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
        y_pred = self.model(x, training=False)
        self.test_accuracy.update_state(y, y_pred)
        self.test_loss.update_state(y, y_pred)
 

    def main(self):
        # self.model.build(input_shape=(None, 32, 32, 3))
        # self.model.summary()
        if args.dataset.lower() == "cifar10":
            (x_train_val, y_train_val), _ = tf.keras.datasets.cifar10.load_data()
        elif args.dataset.lower() == "cifar100":
            (x_train_val, y_train_val), _ = tf.keras.datasets.cifar100.load_data()  
        elif args.dataset.lower() == "mnist":
            (x_train_val, y_train_val), _ = tf.keras.datasets.mnist.load_data()
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported.")     


        x_train, y_train = x_train_val[int(0.1 * len(x_train_val)):], y_train_val[int(0.1 * len(x_train_val)):]
        x_val, y_val = x_train_val[:int(0.1 * len(x_train_val))], y_train_val[:int(0.1 * len(x_train_val))]

        data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        data_train = data_train.shuffle(len(x_train)).batch(args.batch_size)

        data_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        data_val = data_val.batch(args.batch_size)
        
        # data_val = data_train_val.take(int(0.2 * len(x_train))).batch(args.batch_size)
        # data_train = data_train_val.skip(int(0.2 * len(x_train))).batch(args.batch_size)
        
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        checkpoint_dir = os.path.join(DATA_DIR, "checkpoints")
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

        log_dir = os.path.join(DATA_DIR, "logs")
        train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
        val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

        step = 0
        acc_best = 0.0
        for epoch in range(args.epochs):
            for images, labels in data_train.map(map_func=self.preprocess_training_data, 
                                                 num_parallel_calls=tf.data.experimental.AUTOTUNE):
                self.train_step(images, labels)
                step += 1
                if step % 20 == 0:
                    with train_summary_writer.as_default(): 
                        tf.summary.scalar("loss", self.train_loss.result(), step=step)
                        tf.summary.scalar("accuracy", self.train_accuracy.result(), step=step)
                        tf.summary.image("images", images, step=step,max_outputs=4)

                    print(f"Epoch: {epoch+1} \t Step: {step} \t loss = {self.train_loss.result():.6f}\
                        \t acc = {self.train_accuracy.result():.6f}")
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

            for val_images, val_labels in tqdm(data_val.map(map_func=self.preprocess_val_data, 
                                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)):
                self.test_step(val_images, val_labels)
            
            with val_summary_writer.as_default():
                tf.summary.scalar("val_loss", self.test_loss.result(), step=epoch)
                tf.summary.scalar("val_accuracy", self.test_accuracy.result(), step=epoch)
                tf.summary.image("val_images", val_images, step=epoch, max_outputs=4)

            print(f"Epoch: {epoch+1} \t val_loss = {self.test_loss.result():.6f}\
                    \t val_acc = {self.test_accuracy.result():.6f}")
            
            if self.test_accuracy.result() > acc_best:
                ckpt_path = manager.save()
                acc_best = self.test_accuracy.result()
                print(f"Model saved at {ckpt_path}")

            self.test_accuracy.reset_states()
            self.test_loss.reset_states()


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.main()

        




