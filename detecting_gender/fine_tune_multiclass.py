from keras.layers import Activation, Flatten, MaxPooling2D, Dropout
from keras.layers import Convolution2D, ZeroPadding2D, Dense
import keras.backend as K
import tensorflow as tf
from imutils import paths
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from utils import get_random_eraser
import os.path
import argparse
from adabound import AdaBound
import numpy as np
import pandas as pd
from keras_vggface.vggface import VGGFace


matplotlib.use('agg')


def get_optimizer(opt_name, lr, epochs):
    '''
    set optimizer
    '''
    if opt_name == 'sgd':
        return SGD(lr=lr, momentum=0.9, nesterov=False)
    elif opt_name == 'adam':
        return Adam(lr=lr)
    elif opt_name == 'adabound':
        return AdaBound(lr=1e-03, final_lr=0.1, gamma=1e-03, weight_decay=0.,
                        amsbound=False)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam")


def vgg_face(weights_path=None):
    '''
    TODO replace with default function in keras (if model is the same)
    '''
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def get_callbacks(args):
    early_stop = EarlyStopping('val_loss', patience=args.patience, mode='max',
                               verbose=1)
    reduce_lr = ReduceLROnPlateau(
        'val_loss', factor=0.2, patience=int(args.patience/4), min_lr=0.000001,
        verbose=1)
    model_names = os.path.join(
        args.output_path, 'checkpoint-{epoch:02d}-{val_acc:2f}.hdf5')
    model_checkpoint = ModelCheckpoint(model_names,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='auto'
                                       )
    return [early_stop, reduce_lr, model_checkpoint]


def plot_training(H, plotPath):
    plt.style.use("ggplot")
    N = len(H.history["loss"])
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)


def make_generators(args):
    trainPath = os.path.join(args.dataset, "train")
    valPath = os.path.join(args.dataset, "validation")
    testPath = os.path.join(args.dataset, "test")

    if args.eraser:
        preprocessing_function = get_random_eraser(pixel_level=True)
    else:
        preprocessing_function = None

    # initialize the training data augmentation objects
    trainAug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        #zoom_range=[0, 0.2],
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=True,
        preprocessing_function=preprocessing_function,
        fill_mode="nearest"
    )

    valAug = ImageDataGenerator(
        rescale=1.0 / 255
    )

    testAug = ImageDataGenerator(
        rescale=1.0 / 255
    )

    mean = np.array([123.68, 116.779, 103.939], dtype="float32")

    trainAug.mean = mean
    valAug.mean = mean
    testAug.mean = mean

    # initialize the training generator objects
    trainGen = trainAug.flow_from_directory(
        trainPath,
        class_mode="categorical",
        target_size=(args.image_size, args.image_size),
        color_mode="rgb",
        shuffle=True,
        batch_size=args.batch_size)

    # initialize the validation generator
    valGen = valAug.flow_from_directory(
        valPath,
        class_mode="categorical",
        target_size=(args.image_size, args.image_size),
        color_mode="rgb",
        shuffle=False,
        batch_size=args.batch_size)

    # initialize the testing generator
    testGen = testAug.flow_from_directory(
        testPath,
        class_mode="categorical",
        target_size=(args.image_size, args.image_size),
        color_mode="rgb",
        shuffle=False,
        batch_size=args.batch_size)

    print(testGen.class_indices.keys())
    return trainGen, valGen, testGen


def fine_tune(args):

    trainGen, valGen, testGen = make_generators(args)

    trainPath = os.path.join(args.dataset, "train")
    valPath = os.path.join(args.dataset, "validation")
    testPath = os.path.join(args.dataset, "test")

    totalTrain = len(list(paths.list_images(trainPath)))
    totalVal = len(list(paths.list_images(valPath)))
    totalTest = len(list(paths.list_images(testPath)))

    if not args.figures_path:
        os.makedir(args.figures_path)

    unfrozen_plot = os.path.join(args.figures_path, "unfrozen.png")
    warmup_plot = os.path.join(args.figures_path, "warmup.png")

    N_CATEGORIES = 4

    #########################

    print("[INFO] compiling model...")

    hidden_dim = 512

    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('pool5').output
    x = Flatten(name='flatten')(last_layer)
    x = Dense(hidden_dim, activation='relu', name='fc6')(x)
    x = Dense(hidden_dim, activation='relu', name='fc7')(x)
    x = Dropout(0.5, name='dropout')(x)
    out = Dense(N_CATEGORIES, activation='softmax', name='fc8')(x)
    gender_model = Model(vgg_model.input, out)

    for layer in vgg_model.layers:
        layer.trainable = False

    #model = vgg_face('models/vgg_face_weights.h5')

    opt = get_optimizer(args.opt, args.lr, args.n_epochs)

    gender_model.compile(loss="categorical_crossentropy", optimizer=opt,
                         metrics=["accuracy"])

    callbacks = get_callbacks(args)

    # for layer in model.layers[:-7]:
    #    layer.trainable = False

    #################################

    print("[INFO] training head...")
    H = gender_model.fit_generator(
        trainGen,
        verbose=1,
        steps_per_epoch=totalTrain // args.batch_size,
        validation_data=valGen,
        callbacks=callbacks,
        validation_steps=totalVal // args.batch_size,
        epochs=args.n_epochs)

    ##################################

    print("[INFO] evaluating after fine-tuning network head...")
    testGen.reset()
    predIdxs = gender_model.predict_generator(testGen,
                                              steps=(totalTest // args.batch_size) + 1)
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testGen.classes, predIdxs,
                                target_names=testGen.class_indices.keys()))
    plot_training(H, warmup_plot)

    trainGen.reset()
    valGen.reset()

    for layer in vgg_model.layers[15:]:
        layer.trainable = True

    for layer in vgg_model.layers:
        print("{}: {}".format(layer, layer.trainable))

    print('re-compiling model')
    gender_model.compile(loss="categorical_crossentropy", optimizer=opt,
                         metrics=["accuracy"])

    H = gender_model.fit_generator(
        trainGen,
        verbose=1,
        steps_per_epoch=totalTrain // args.batch_size,
        validation_data=valGen,
        callbacks=callbacks,
        validation_steps=totalVal // args.batch_size,
        epochs=20)

    print("[INFO] evaluating after fine-tuning network head...")
    testGen.reset()
    predIdxs = gender_model.predict_generator(testGen,
                                              steps=(totalTest // args.batch_size) + 1)
    predIdxs = np.argmax(predIdxs, axis=1)

    print(classification_report(testGen.classes, predIdxs,
                                target_names=testGen.class_indices.keys()))
    plot_training(H, unfrozen_plot)

    print('saving model')

    pd.DataFrame(H.history).to_hdf(os.path.join("history.h5"), "history")

    gender_model.save(args.model_path)

    # # serialize the model to disk
    # print("[INFO] serializing network...")
    # pd.DataFrame(H.history).to_hdf("history.h5")
    # gender_model.save(args.model_path)


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tuning vgg16 for gender estimation in historical adverts.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size")
    parser.add_argument("--n_epochs", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="initial learning rate")
    parser.add_argument("--opt", type=str, default="adabound",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--patience", type=int, default=6,
                        help="Patience for callbacks")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input size of images")
    parser.add_argument("--eraser", action="store_true")
    parser.add_argument("--dataset", type=str, default="../data/gender_2")
    parser.add_argument("--aug", action="store_true",
                        help="use data augmentation if set true")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--figures_path", type=str, default="figures",
                        help="path for figures")
    parser.add_argument("--model_path", type=str, default="multi_model.h5",
                        help="model dir")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['KERAS_BACKEND'] = 'tensorflow'

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    args = get_args()
    fine_tune(args)
