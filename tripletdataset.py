import tensorflow as tf
import numpy as np
import os
from skimage.filters import gaussian

class TripletDataset:

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    batch_size = 8
    impaths = []
    images = []
    labels = []

    def __init__(self, root_path, set_codes, batch_size):
        self.batch_size = batch_size
        self.impaths = self.get_card_paths(root_path, set_codes)
        self.labels = [i for i in range(len(self.impaths))]
        np.random.shuffle(self.labels)
        np.random.shuffle(self.impaths)
        assert len(self.impaths) >= 100, 'Too little data. Please include at least 100 examples'
        print('loading {} images'.format(len(self.impaths)))
        self.images = [self.load_image(img) for img in self.impaths]
        self.train_images = self.images[:int(len(self.images)*0.9)]
        self.train_labels = self.labels[:len(self.train_images)]
        self.test_images = self.images[len(self.train_images):]
        self.test_labels = self.labels[len(self.train_images):]
        assert len(self.test_images) + len(self.train_images) == len(self.images), \
            'Some data points seem to have been dropped: {}+{} not {}'.format(
                len(self.test_images), len(self.train_images), len(self.impaths)
                )

    def get_card_paths(self, root_path, set_codes):
        card_paths = []
        print(root_path, set_codes)
        for s in set_codes:
            assert os.path.exists(os.path.join(root_path, s)), \
                'Error: path {} does not exist'.format(os.path.join(root_path, s))
            set_path = os.listdir(os.path.join(root_path, s))
            for card in set_path:
                card_paths.append(os.path.join(root_path, s, card))
        return card_paths

    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [385, 275])
        return img

    def preprocess(self, image):
        img = tf.image.random_brightness(image, max_delta=0.6)
        img = tf.image.random_contrast(img, 0.2, 3.0)
        img = tf.image.random_jpeg_quality(img, 20, 80)
        img = gaussian(img, sigma=np.random.choice(range(1,4)))
        img = tf.clip_by_value(img, 0.0, 1.0)
        img = tf.image.random_crop(img, [350, 250, 3])
        return img
    
    def get_triplet(self, train=True):
        img_a = []
        img_b = []
        if not train:
            idxs = np.random.randint(0, len(self.test_images), size=2)
            img_a = self.test_images[idxs[0]]
            img_b = self.test_images[idxs[1]]
        else:
            idxs = np.random.randint(0, len(self.train_images), size=2)
            img_a = self.train_images[idxs[0]]
            img_b = self.train_images[idxs[1]]
        anker = self.preprocess(img_a)
        positive = self.preprocess(img_a)
        negative = self.preprocess(img_b)
        return anker, positive, negative

    def get_triplet_batch(self, train=True):
        ankers, positives, negatives = [], [], []
        for i in range(self.batch_size):
            a, p, n = self.get_triplet(train)
            ankers.append(a)
            positives.append(p)
            negatives.append(n)
        return ankers, positives, negatives
