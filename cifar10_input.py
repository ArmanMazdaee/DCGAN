import numpy as np

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_NUM_CHANNELS = 3


def load_cifar10(file_paths, batch_size):
    images = []
    for path in file_paths:
        images.append(np.fromfile(path, np.uint8))
    images = np.array(images, np.float32)
    images = images.reshape(-1, IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_NUM_CHANNELS+1)
    images = images[:, 1:]
    images = images.reshape(-1, IMAGE_NUM_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT)
    images = images.transpose(0, 2, 3, 1)
    images = (images - 127.5) / 127.5

    indexes = np.arange(len(images))
    image_batch = []
    while True:
        np.random.shuffle(indexes)
        for i in indexes:
            image_batch.append(images[i])
            if len(image_batch) == batch_size:
                yield np.array(image_batch)
                image_batch = []
