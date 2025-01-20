# USAGE
# python train.py --checkpoints output/checkpoints
# python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch25.hdf5 --start-epoch 25
import matplotlib
matplotlib.use("Agg")

from config import tiny_imagenet_config as config
from MCVDT.preprocessing import ImageToArrayPreprocessor
from MCVDT.preprocessing import SimplePreprocessor
from MCVDT.preprocessing import MeanPreprocessor
from MCVDT.callbacks import EpochCheckpoint
from MCVDT.callbacks import TrainingMonitor
from MCVDT.io import HDF5DatasetGenerator
from MCVDT.nn.conv import ResNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_value as K
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
                help="path to specific model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug,
                                preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64,
                              preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

if args["model"] is None:
    print("[INFO] compiling model...")
    model = ResNet.build(64, 64, 3, config.NUM_CLASSES,
                         (3, 4, 6), (64, 128, 256, 512), reg=0.0005, dataset="tiny_imagenet")
    opt = SGD(lr=1e-1, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    print("[INFO] old learning rate: {}".format(
        K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-2)
    print("[INFO] new learning rate: {}".format(
        K.get_value(model.optimizer.lr)))

callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5,
                    startAt=args["start_epoch"]),
    TrainingMonitor(config.FIG_PATH, jsonPath=config.JSON_PATH,
                    startAt=args["start_epoch"])
]

model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 64,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 64,
    epochs=0,
    max_queue_size=10,
    callbacks=callbacks, verbose=1)

trainGen.close()
valGen.close()
