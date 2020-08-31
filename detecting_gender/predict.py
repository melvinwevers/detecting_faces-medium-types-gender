from adabound import AdaBound
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import CustomObjectScope
import os.path
import tensorflow as tf
import keras.backend as K
import numpy as np
import argparse
from sklearn.metrics import classification_report


def make_generators(args):
    testPath = os.path.join(args.dataset, "test/1990")

    testAug = ImageDataGenerator(
        rescale=1.0 / 255
    )

    mean = np.array([123.68, 116.779, 103.939], dtype="float32")

    testAug.mean = mean

    # initialize the testing generator
    testGen = testAug.flow_from_directory(
        testPath,
        class_mode="binary",
        target_size=(args.image_size, args.image_size),
        color_mode="rgb",
        shuffle=False,
        batch_size=args.batch_size)

    print(testGen.class_indices.keys())
    return testGen


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
    parser.add_argument("--dataset", type=str, default="../data/gender_3")
    parser.add_argument("--aug", action="store_true",
                        help="use data augmentation if set true")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--figures_path", type=str, default="figures",
                        help="path for figures")
    parser.add_argument("--model_path", type=str, default="model.h5",
                        help="model dir")

    args = parser.parse_args()
    return args


def evaluate(args):
    testGen = make_generators(args)

    with CustomObjectScope({'AdaBound': AdaBound()}):
        model = load_model('binary_model.h5')

    # print(testGen.classes)

    predIdxs = model.predict_generator(testGen, steps=(
        testGen.samples // args.batch_size) + 1, verbose=1)

    predIdxs = predIdxs > 0.5
    # print(predIdxs)

    print(classification_report(testGen.classes, predIdxs,
                                target_names=testGen.class_indices.keys()))


if __name__ == '__main__':
    os.environ['KERAS_BACKEND'] = 'tensorflow'

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    args = get_args()
    evaluate(args)
