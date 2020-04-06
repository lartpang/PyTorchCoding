import os
import random
import types
from random import shuffle

import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class ExternalInputIterator(object):
    def __init__(self, root, batch_size, random_shuffle=True):
        self.source_path = os.path.join(root, 'Image')
        self.mask_path = os.path.join(root, 'Mask')
        self.files = [x.split('.')[0] for x in os.listdir(self.source_path)]
        self.batch_size = batch_size
        if random_shuffle:
            shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        try:
            images = []
            masks = []
            for _ in range(self.batch_size):
                img_path = os.path.join(self.source_path, self.files[self.i] + '.jpg')
                gt_path = os.path.join(self.mask_path, self.files[self.i] + '.png')

                with open(img_path, 'rb') as img:
                    images.append(np.frombuffer(img.read(), dtype=np.uint8))
                with open(gt_path, 'rb') as gt:
                    masks.append(np.frombuffer(gt.read(), dtype=np.uint8))

                self.i = (self.i + 1)
            return (images, masks)
        except:
            self.i = 0
            raise StopIteration

    next = __next__


class ImagePipeline(Pipeline):
    def __init__(
        self, imageset_dir, image_size, random_shuffle, batch_size=4, num_threads=2, device_id=0
    ):
        super(ImagePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.imageset_dir = imageset_dir
        self.random_shuffle = random_shuffle

        eii = ExternalInputIterator(
            root=self.imageset_dir, batch_size=self.batch_size, random_shuffle=self.random_shuffle
        )
        self.iterator = iter(eii)
        self.num_inputs = len(self.iterator.files)

        self.input_image = ops.ExternalSource()
        self.input_mask = ops.ExternalSource()

        self.decode_image = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.decode_mask = ops.ImageDecoder(device="mixed", output_type=types.GRAY)

        # The rest of pre-processing is done on the GPU
        self.res = ops.Resize(device="gpu", resize_x=image_size, resize_y=image_size)
        self.flip = ops.Flip(device="gpu", horizontal=1, vertical=0)

        rotate_degree = random.random() * 2 * 10 - 10
        self.rotate_image = ops.Rotate(
            device="gpu", angle=rotate_degree, interp_type=types.DALIInterpType.INTERP_LINEAR
        )
        self.rotate_mask = ops.Rotate(
            device="gpu", angle=rotate_degree, interp_type=types.DALIInterpType.INTERP_NN
        )

        self.cmnp_image = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            image_type=types.RGB,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
        )
        self.cmnp_mask = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            image_type=types.GRAY,
            mean=[0],
            std=[255]
        )

    # epoch_size = number of (image, mask) image pairs in the dataset
    def epoch_size(self, name=None):
        return self.num_inputs

    def define_graph(self):
        self.images = self.input_image(name="images")
        self.masks = self.input_mask(name="masks")

        images = self.decode_image(self.images)
        masks = self.decode_mask(self.masks)

        output_image = self.res(images)
        output_mask = self.res(masks)

        if random.random() < 0.5:
            output_image = self.flip(output_image)
            output_mask = self.flip(output_mask)

        output_image = self.rotate_image(output_image)
        output_mask = self.rotate_mask(output_mask)

        output_image = self.cmnp_image(output_image)
        output_mask = self.cmnp_mask(output_mask)

        return (output_image, output_mask)

    def iter_setup(self):
        try:
            (images, masks) = self.iterator.next()
            self.feed_input(self.images, images)
            self.feed_input(self.masks, masks)
        except StopIteration:
            self.iterator = iter(
                ExternalInputIterator(
                    root=self.imageset_dir,
                    batch_size=self.batch_size,
                    random_shuffle=self.random_shuffle
                )
            )
            raise StopIteration


if __name__ == '__main__':
    datapath = 'training_set'
    gpu_id = 0
    batch_size = 5

    train_pipe = ImagePipeline(
        imageset_dir='/home/erti/Datasets/RGBSaliency/DUTS/Train',
        image_size=128,
        random_shuffle=False,
        batch_size=batch_size
    )
    m_train = train_pipe.epoch_size()
    print("Size of the training set: ", m_train)
    train_pipe_loader = DALIGenericIterator(
        pipelines=train_pipe,
        output_map=["images", "masks"],
        size=m_train,
        auto_reset=True,
        fill_last_batch=False,
        last_batch_padded=True
    )
    # 只有batchsize可以整除整个数据集的长度的时候，才能触发shuffle
    for j in range(5):
        print(train_pipe.iterator.files)
        for i, train_data in enumerate(train_pipe_loader):
            train_inputs = train_data[0]['images']
            train_labels = train_data[0]['masks']
            print(train_inputs.max(), train_inputs.min(), train_labels.max(), train_labels.min())
