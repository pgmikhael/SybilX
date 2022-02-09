import torch
import os
import sys
import os.path
import warnings
from sybilx.datasets.utils import get_scaled_annotation_mask, IMG_PAD_TOKEN
from sybilx.augmentations.basic import ComposeAug
import numpy as np
from abc import ABCMeta, abstractmethod
import hashlib

CACHED_FILES_EXT = ".png"
DEFAULT_CACHE_DIR = "default/"

CORUPTED_FILE_ERR = (
    "WARNING! Error processing file from cache - removed file from cache. Error: {}"
)


def md5(key):
    """
    returns a hashed with md5 string of the key
    """
    return hashlib.md5(key.encode()).hexdigest()


def split_augmentations_by_cache(augmentations):
    """
    Given a list of augmentations, returns a list of tuples. Each tuple
    contains a caching key of the augmentations up to the spiltting point,
    and a list of augmentations that should be applied afterwards.

    split_augmentations will contain all possible splits by cachable augmentations,
    ordered from latest possible one to the former ones.
    The last tuple will have all augmentations.

    Note - splitting will be done for indexes that all augmentations up to them are
    cachable.
    """
    # list of (cache key, post augmentations)
    split_augmentations = []
    split_augmentations.append((DEFAULT_CACHE_DIR, augmentations))
    all_prev_cachable = True
    key = ""
    for ind, trans in enumerate(augmentations):

        # check trans.cachable() first separately to save run time
        if not all_prev_cachable or not trans.cachable():
            all_prev_cachable = False
        else:
            key += trans.caching_keys()
            post_augmentations = (
                augmentations[ind + 1 :] if ind < len(augmentations) else []
            )
            split_augmentations.append((key, post_augmentations))

    return list(reversed(split_augmentations))


def apply_augmentations_and_cache(
    image,
    mask,
    additional,
    img_path,
    augmentations,
    cache,
    cache_full_size=False,
    base_key="",
):
    """
    Loads the image by its absolute path and apply the augmentations one
    by one (similar to what the composed one is doing).  All first cachable
    transformer's output is cached (until reaching a non cachable one).
    """
    if cache_full_size and not cache.exists(img_path, DEFAULT_CACHE_DIR):
        cache.add(img_path, DEFAULT_CACHE_DIR, image)

    all_prev_cachable = True
    key = base_key
    for ind, trans in enumerate(augmentations):
        image, mask = trans(image, mask, additional)
        if not all_prev_cachable or not trans.cachable():
            all_prev_cachable = False
        else:
            key += trans.caching_keys()
            cache.add(img_path, key, image)

    return image, mask


class cache:
    def __init__(self, path, extension=CACHED_FILES_EXT):
        if not os.path.exists(path):
            os.makedirs(path)

        self.cache_dir = path
        self.files_extension = extension
        if ".npy" != extension:
            self.files_extension += ".npy"

    def _file_dir(self, attr_key, par_dir):
        return os.path.join(self.cache_dir, attr_key, par_dir)

    def _file_path(self, attr_key, par_dir, hashed_key):
        return os.path.join(
            self.cache_dir, attr_key, par_dir, hashed_key + self.files_extension
        )

    def _parent_dir(self, path):
        return os.path.basename(os.path.dirname(path))

    def exists(self, image_path, attr_key):
        hashed_key = md5(image_path)
        par_dir = self._parent_dir(image_path)
        return os.path.isfile(self._file_path(attr_key, par_dir, hashed_key))

    def get(self, image_path, attr_key):
        hashed_key = md5(image_path)
        par_dir = self._parent_dir(image_path)
        return np.load(self._file_path(attr_key, par_dir, hashed_key))

    def add(self, image_path, attr_key, image):
        hashed_key = md5(image_path)
        par_dir = self._parent_dir(image_path)
        file_dir = self._file_dir(attr_key, par_dir)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        np.save(self._file_path(attr_key, par_dir, hashed_key), image)

    def rem(self, image_path, attr_key):
        hashed_key = md5(image_path)
        par_dir = self._parent_dir(image_path)
        try:
            os.remove(self._file_path(attr_key, par_dir, hashed_key))
        # Don't raise error if file not exists.
        except OSError:
            pass


class abstract_loader:
    __metaclass__ = ABCMeta

    def __init__(self, cache_path, augmentations, args):
        self.pad_token = IMG_PAD_TOKEN
        self.augmentations = augmentations
        self.args = args
        if cache_path is not None:
            self.use_cache = True
            self.cache = cache(cache_path, self.cached_extension)
            self.split_augmentations = split_augmentations_by_cache(augmentations)
        else:
            self.use_cache = False
            self.composed_all_augmentations = ComposeAug(augmentations)

    @abstractmethod
    def configure_path(self, path, additional, sample):
        pass

    @abstractmethod
    def load_input(self, path, additional):
        pass

    @property
    @abstractmethod
    def cached_extension(self):
        pass

    @property
    @abstractmethod
    def apply_augmentations(self):
        return True

    def get_image(self, path, additional, sample):
        """
        Returns a transformed image by its absolute path.
        If cache is used - transformed image will be loaded if available,
        and saved to cache if not.
        """
        image_path = self.configure_path(path, additional, sample)
        mask = (
            get_scaled_annotation_mask(additional, self.args)
            if self.args.use_annotations
            else None
        )
        if image_path == self.pad_token:
            shape = (self.args.num_chan, self.args.img_size[0], self.args.img_size[1])
            image = torch.zeros(*shape)
            mask = (
                torch.from_numpy(mask).unsqueeze(0)
                if self.args.use_annotations
                else None
            )
            return image, mask

        if not self.use_cache:
            image = self.load_input(image_path, additional)
            # hidden loaders typically do not use augmentation
            if self.apply_augmentations:
                image, mask = self.composed_all_augmentations(image, mask, additional)
            return image, mask

        for key, post_augmentations in self.split_augmentations:
            base_key = (
                DEFAULT_CACHE_DIR + key
                if key != DEFAULT_CACHE_DIR
                else DEFAULT_CACHE_DIR
            )
            if self.cache.exists(image_path, base_key):
                try:
                    image = self.cache.get(image_path, base_key)
                    if self.apply_augmentations:
                        image, mask = apply_augmentations_and_cache(
                            image,
                            mask,
                            additional,
                            image_path,
                            post_augmentations,
                            self.cache,
                            cache_full_size=self.args.cache_full_img,
                            base_key=base_key,
                        )
                    return image, mask
                except Exception as e:
                    print(e)
                    hashed_key = md5(image_path)
                    par_dir = self.cache._parent_dir(image_path)
                    corrupted_file = self.cache._file_path(key, par_dir, hashed_key)
                    warnings.warn(CORUPTED_FILE_ERR.format(sys.exc_info()[0]))
                    self.cache.rem(image_path, key)
        all_augmentations = self.split_augmentations[-1][1]
        image = self.load_input(image_path, additional)
        if self.apply_augmentations:
            image, mask = apply_augmentations_and_cache(
                image,
                mask,
                additional,
                image_path,
                all_augmentations,
                self.cache,
                cache_full_size=self.args.cache_full_img,
                base_key=key,
            )

        return image, mask

    def get_images(self, paths, additionals, sample):
        """
        Returns a stack of transformed images by their absolute paths.
        If cache is used - transformed images will be loaded if available,
        and saved to cache if not.
        """

        sample_fixed_seed = np.random.randint(0, 2**32 - 1)
        additionals += [{}] * (len(paths) - len(additionals))
        for i, addit in enumerate(additionals):
            if addit is None:
                additionals[i] = {}
            if self.args.fix_seed_for_multi_image_augmentations and (
                "seed" not in additionals[i]
            ):
                additionals[i]["seed"] = sample_fixed_seed

        # get images for multi image input
        images_masks = [
            self.get_image(path, additional, sample)
            for path, additional in zip(paths, additionals)
        ]
        images, masks = [i[0] for i in images_masks], [i[1] for i in images_masks]

        images = self.reshape_images(images)
        masks = self.reshape_images(masks) if self.args.use_annotations else None

        return images, masks

    def reshape_images(self, images):
        images = [im.unsqueeze(0) for im in images]
        images = torch.cat(images, dim=0)
        # Convert from (T, C, H, W) to (C, T, H, W)
        images = images.permute(1, 0, 2, 3)
        return images
