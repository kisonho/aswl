# import system modules
import argparse
from typing import List, Tuple

# import numpy modules
import numpy as np

# import tensorflow modules
import tensorflow as tf
# import tensorflow_datasets as TFDataset
from tensorflow import keras as K

# import core modules
from aswl.models import CompressibleModel, ModelBuilder
from aswl.optimizers import CompressionOptimizer
import aswl.metrics as Metrics

# define training function
def train(m: str, epochs: int=10, batch_size: int=32, weights_lr: float=0.001, weight_decay: float=0, attention_decay: float=0, initial_epoch: int=0, alpha: float=1, experiment: str='test'):
    # initialize directory
    data_dir: str = str('Data/%s/' % experiment).replace("/", "\\")
    ckpt_dir: str = str('checkpoints/%s/' % experiment).replace("/", "\\")

    # keras datasets
    # initialize dataset
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255 
    x_test = x_test.astype("float32") / 255 
    x_train_mean: list = np.mean(x_train, axis=0)
    x_train -= np.mean(x_train_mean, axis=0)
    x_test -= np.mean(x_train_mean, axis=0)
    num_classes: int = 10
    y_train = K.utils.to_categorical(y_train, num_classes, dtype="int32")
    y_test = K.utils.to_categorical(y_test, num_classes, dtype="int32")
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    steps_per_epoch: int = int(len(x_train) / batch_size)

    '''
    #  calculate smaple size for regularized datasets
    sample_size: Tuple[int, int, int] = (info.splits['train'].num_examples, info.splits['test'].num_examples, info.splits['test'].num_examples)
    input_shape = info.features.get_tensor_info()['image'].shape
    '''

    # print input shape
    tf.print('Dataset input shape:', input_shape)
    
    # initialize lr
    # scheduled_weights_lr = K.optimizers.schedules.ExponentialDecay(weights_lr, steps_per_epoch, weight_lr_decay, staircase=True)
    # scheduled_attentions_lr = K.optimizers.schedules.ExponentialDecay(attentions_lr, steps_per_epoch, attention_lr_decay, staircase=True)
    # scheduled_weights_lr = K.experimental.CosineDecay(weights_lr, steps_per_epoch)
    # scheduled_attentions_lr = K.experimental.CosineDecay(weights_lr, steps_per_epoch)

    # initialize model builder
    model_builder: ModelBuilder = ModelBuilder(input_shape, num_classes, m, weight_decay=weight_decay, attention_decay=attention_decay, pruning_factor=alpha)
    
    # initialize model
    model: CompressibleModel = model_builder.build()

    # initialize optimizer
    momentum: float = 0.9
    weights_optimizer = K.optimizers.SGD(learning_rate=weights_lr, momentum=momentum)
    attentions_optimizer = K.optimizers.SGD(learning_rate=weights_lr, momentum=momentum)
    '''
    weights_optimizer = K.optimizers.Adam(learning_rate=scheduled_weights_lr)
    attentions_optimizer = K.optimizers.Adam(learning_rate=scheduled_attentions_lr)
    '''
    optimizer = CompressionOptimizer(weights_optimizer, attentions_optimizer=attentions_optimizer, alpha=alpha)

    # initialize loss and accuracy
    loss: K.losses.SparseCategoricalCrossentropy = K.losses.SparseCategoricalCrossentropy(name='loss')
    accuracy = K.metrics.SparseCategoricalAccuracy(name='accuracy')
    average_pruning_ratio = Metrics.AveragePruningRatio(model, alpha)
    
    # initialize model
    model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy, average_pruning_ratio])
    model.summary()

    # restore checkpoint
    if initial_epoch > 0:
        latest_ckpt = tf.train.latest_checkpoint(ckpt_dir + "last".replace("/","\\"))
        model.load_weights(latest_ckpt)
        model.evaluate(x_test, y_test, batch_size=batch_size)
    
    # initialize uncompressed weights in optimizer
    optimizer.initialize_uncompressed_weights(model)

    # learning rate schedule
    def lr_schedule(epoch: int, lr: float) -> float:
        lr = weights_lr
        if epoch >= 48000 / steps_per_epoch:
            lr *= 1e-2
        elif epoch >= 32000 / steps_per_epoch:
            lr *= 0.1
        return lr
            
    # training callbacks
    tensorboard_callback: K.callbacks.TensorBoard = K.callbacks.TensorBoard(log_dir=data_dir, update_freq='batch')
    checkpoint_callback: K.callbacks.ModelCheckpoint = K.callbacks.ModelCheckpoint(filepath=ckpt_dir + "last/ckpt".replace("/","\\"), save_weights_only=True)
    best_loss_checkpoint_callback: K.callbacks.ModelCheckpoint = K.callbacks.ModelCheckpoint(filepath=ckpt_dir + "best_loss/ckpt".replace("/","\\"), save_weights_only=True, save_best_only=True)
    best_acc_checkpoint_callback: K.callbacks.ModelCheckpoint = K.callbacks.ModelCheckpoint(filepath=ckpt_dir + "best_accuracy/ckpt".replace("/","\\"), save_weights_only=True, save_best_only=True, monitor='val_accuracy', mode='max')
    lr_schedule_callback: K.callbacks.LearningRateScheduler = K.callbacks.LearningRateScheduler(lr_schedule)
    callbacks: List[K.callbacks.Callback] = [tensorboard_callback, checkpoint_callback, best_loss_checkpoint_callback, best_acc_checkpoint_callback, lr_schedule_callback]

    # print initial attentions
    tf.print('Initial attentions:', model.attentions)
    
    # preprocessing generator
    data_generator: K.preprocessing.image.ImageDataGenerator = K.preprocessing.image.ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.1, height_shift_range=0.1, shear_range=0., zoom_range=0., channel_shift_range=0., fill_mode='nearest', cval=0., horizontal_flip=True, vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.1)
    data_generator.fit(x_train)

    # fit model
    model.fit(data_generator.flow(x_train, y_train, batch_size=batch_size, subset='training'), epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks, validation_data=data_generator.flow(x_train, y_train, batch_size=batch_size, subset='validation'), workers=4)

    # load best ckpt
    best_ckpt = tf.train.latest_checkpoint(ckpt_dir + 'best_accuracy/'.replace("/", "\\"))
    model.load_weights(best_ckpt)

    # evaluate
    model.evaluate(x_test, y_test, batch_size=batch_size)
    
    # print final attentions
    attentions: tf.Tensor = tf.convert_to_tensor(model.attentions)
    tf.print('Final attentions:', attentions)

    # print attention loss
    if model.attention_regularizer is not None:
        # calculate attention loss
        tf.print('Attention loss:', model.attention_regularizer(attentions))

    # calculate final pruning ratios
    pruning_ratios: tf.Tensor = tf.pow((1 - attentions), alpha)
    tf.print('Final pruning ratio:', pruning_ratios)

    # save to csv file
    np.savetxt(data_dir + "attentions.csv", np.array(attentions), delimiter=",")
    np.savetxt(data_dir + "pruning_ratios.csv", np.array(pruning_ratios), delimiter=",")
    return model

# main function
if __name__=="__main__":
    # add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The name of base model. Either VGG16, MobileNetV2, ResNet56, or ResNet50V2.")
    parser.add_argument("--weights_lr", type=float, default=0.001, help="Initial weights learning rate. Default is 0.001.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay of the kernel regularizer, use negative numbers for default regularizer set in keras applications. Default is 0.")
    parser.add_argument("--attention_decay", type=float, default=0, help="Attention decay of the attention regularizer. Default is 0.")
    parser.add_argument("--alpha", type=float, default=1, help="Pruning factor. Default is 1.")
    parser.add_argument("--initial_epoch", type=int, default=0, help="The initial epoch index where training started. Default is 0.")
    parser.add_argument("--epochs", type=int, default=10, help="The epochs to train. Default is 10.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size of the dataset. Default is 32")
    parser.add_argument("--experiment", type=str, default='test', help="Experiment name. Default is \'test\'.")

    # get arguments
    args = parser.parse_args()
    m: str = args.model
    weights_lr: float = args.weights_lr
    weight_decay: float = args.weight_decay
    attention_decay: float = args.attention_decay
    alpha: float = args.alpha
    initial_epoch: int = args.initial_epoch
    epochs: int = args.epochs
    batch: int = args.batch_size
    experiment: str = args.experiment

    # train model
    train(m, epochs=epochs, batch_size=batch, weights_lr=weights_lr, weight_decay=weight_decay, attention_decay=attention_decay, initial_epoch=initial_epoch, alpha=alpha, experiment=experiment)
else:
    print('Function %s, import only. Use train method to train model.' % __name__)
