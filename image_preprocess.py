import numpy as np
import tensorflow as tf
import cv2
import os

# point to proper animal dirs
DOGS_PATH = '../dogs'
CATS_PATH = '../cats'


def fetch_resized_animals(base_path):
    image_paths = os.listdir(base_path)
    animals = [cv2.imread(base_path + os.sep + animal, cv2.IMREAD_GRAYSCALE) for animal in image_paths]
    return [cv2.resize(animal, (37, 37)) / 255. for animal in animals if animal is not None]


def load_datasets():
    if not os.path.exists('dogs.npy'):
        dogs = np.array(
            [[np.expand_dims(dog, axis=2), np.array([1, 0], np.float)] for dog in fetch_resized_animals(DOGS_PATH)])
        np.save('dogs.npy', dogs)
    else:
        dogs = np.load('dogs.npy', allow_pickle=True)

    if not os.path.exists('cats.npy'):
        cats = np.array(
            [[np.expand_dims(cat, axis=2), np.array([0, 1], np.float)] for cat in fetch_resized_animals(CATS_PATH)])
        np.save('cats.npy', cats)
    else:
        cats = np.load('cats.npy', allow_pickle=True)
    data = np.concatenate((dogs, cats))
    return data


def bootstrap_data_preprocess():
    data = load_datasets()
    x_train = []
    y_train = []
    for datum in data:
        x_train.append(datum[0])
        y_train.append(datum[1])

    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True, rotation_range=15,
                                                                     width_shift_range=0.3, height_shift_range=0.1,
                                                                     horizontal_flip=True, zca_whitening=True)
    x = np.array(x_train, dtype=np.float32)
    y = np.array(y_train, dtype=np.float32)
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    # data augmentation to increase dataset size
    data_generator.fit(x)
    aug_batches_iter = data_generator.flow(x, y, batch_size=16)
    train_data = tf.data.Dataset.from_tensor_slices((x_train[-500:], y_train[-500:])).shuffle(100000).batch(16)
    test_data = tf.data.Dataset.from_tensor_slices((x_train[:-500], y_train[:-500])).shuffle(20000).batch(16)
    i = 0
    for aug_batch in aug_batches_iter:
        train_data.concatenate(tf.data.Dataset.from_tensor_slices(aug_batch))
        if i == 100:
            break
        i += 1

    return train_data, test_data
