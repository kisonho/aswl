# import system modules
import argparse, os
from typing import List, Tuple

# import numpy modules
import numpy as np

# import tensorflow modules
import tensorflow as tf
from keras import optimizers, preprocessing, utils
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.datasets import cifar10

# import core modules
from aswl import metrics
from applications import ModelBuilder

# main function
if __name__=="__main__":
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ResNet56", help="The name of base model. Either VGG16, MobileNetV2, ResNet56.")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate. Default is 0.1.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay of the kernel regularizer, use negative numbers for default regularizer set in keras applications. Default is 1e-4.")
    parser.add_argument("--attention_decay", type=float, default=5, help="Attention decay of the attention regularizer. Default is 5.")
    parser.add_argument("--alpha", type=float, default=1, help="Pruning factor. Default is 1.")
    parser.add_argument("--initial_epoch", type=int, default=0, help="The initial epoch index where training started. Default is 0.")
    parser.add_argument("--epochs", type=int, default=10, help="The epochs to train. Default is 10.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size of the dataset. Default is 32")
    parser.add_argument("--gpu_num", type=int, default=1, help="GPU used for training. Default is 1.")
    parser.add_argument("--experiment", type=str, default='test', help="Experiment name. Default is \'test\'.")

    # get arguments
    args = parser.parse_args()
    m: str = args.model
    initial_lr: float = args.lr
    weight_decay: float = args.weight_decay
    attention_decay: float = args.attention_decay
    alpha: float = args.alpha
    initial_epoch: int = args.initial_epoch
    epochs: int = args.epochs
    batch_size: int = args.batch_size
    gpu_num: int = args.gpu_num
    experiment: str = args.experiment

    # initialize routes
    data_dir: str = os.path.normpath('Data/%s/' % experiment)
    ckpt_dir: str = os.path.normpath('checkpoints/%s/' % experiment)

    # initialize dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255 
    x_test = x_test.astype("float32") / 255 
    x_train_mean: list = np.mean(x_train, axis=0)
    x_train -= np.mean(x_train_mean, axis=0)
    x_test -= np.mean(x_train_mean, axis=0)
    num_classes: int = 10
    y_train = utils.to_categorical(y_train, num_classes, dtype="int32")
    y_test = utils.to_categorical(y_test, num_classes, dtype="int32")
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    steps_per_epoch: int = int(len(x_train) / batch_size)
    
    # initialize device list
    device_list = ["/gpu:" + str(i) for i in range(gpu_num)]

    # initialize mirrored strategy
    mirrored_strategy: tf.distribute.MirroredStrategy = tf.distribute.MirroredStrategy(devices=device_list)

    # mirrored strategy scope
    with mirrored_strategy.scope():
        # initialize model builder
        model_builder: ModelBuilder = ModelBuilder(input_shape, num_classes, m, weight_decay=weight_decay, attention_decay=attention_decay, pruning_factor=alpha)
        
        # initialize model
        model = model_builder.build()

        # initialize optimizer
        momentum: float = 0.9
        optimizer = optimizers.SGD(learning_rate=initial_lr, momentum=momentum)
        '''
        optimizer = optimizers.Adam(learning_rate=initial_lr)
        '''

        # initialize loss and accuracy
        average_pruning_ratio = metrics.AveragePruningRatio(model, alpha, gpu_num_reduce=gpu_num, name='pruning_ratio')
        
        # initialize model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', average_pruning_ratio])
        model.summary()

        # restore checkpoint
        if initial_epoch > 0:
            latest_ckpt = tf.train.latest_checkpoint(os.path.join(ckpt_dir, "last"))
            model.load_weights(latest_ckpt)
            model.evaluate(x_test, y_test, batch_size=batch_size)
            
    # learning rate schedule
    def lr_schedule(epoch: int, lr: float) -> float:
        lr = initial_lr
        if epoch >= 48000 / steps_per_epoch:
            lr *= 1e-2
        elif epoch >= 32000 / steps_per_epoch:
            lr *= 0.1
        return lr
    
    # training callbacks
    tensorboard_callback: TensorBoard = TensorBoard(log_dir=data_dir)
    checkpoint_callback: ModelCheckpoint = ModelCheckpoint(filepath=os.path.join(ckpt_dir, "last", "ckpt"), save_weights_only=True)
    best_loss_checkpoint_callback: ModelCheckpoint = ModelCheckpoint(filepath=os.path.join(ckpt_dir, "best_loss", "ckpt"), save_weights_only=True, save_best_only=True)
    best_acc_checkpoint_callback: ModelCheckpoint = ModelCheckpoint(filepath=os.path.join(ckpt_dir, "best_accuracy", "ckpt"), save_weights_only=True, save_best_only=True, monitor='val_accuracy', mode='max')
    lr_schedule_callback: LearningRateScheduler = LearningRateScheduler(lr_schedule)
    callbacks: List[Callback] = [tensorboard_callback, checkpoint_callback, best_loss_checkpoint_callback, best_acc_checkpoint_callback, lr_schedule_callback]

    # preprocessing generator
    data_generator: preprocessing.image.ImageDataGenerator = preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.1, height_shift_range=0.1, shear_range=0., zoom_range=0., channel_shift_range=0., fill_mode='nearest', cval=0., horizontal_flip=True, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.1)
    data_generator.fit(x_train)
    
    # train model
    model.fit(data_generator.flow(x_train, y_train, batch_size=batch_size, subset='training'), epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks, validation_data=data_generator.flow(x_train, y_train, batch_size=batch_size, subset='validation'), workers=4)

    # load best ckpt
    with mirrored_strategy.scope():
        best_ckpt = tf.train.latest_checkpoint(os.path.join(ckpt_dir, 'best_accuracy', 'ckpt'))
        model.load_weights(best_ckpt)

        # evaluate
        model.evaluate(x_test, y_test, batch_size=batch_size)
