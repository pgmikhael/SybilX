from collections import Counter
from random import shuffle

import numpy as np
from torch.utils import data
from tqdm import tqdm

from sybilx.utils.registry import register_object

METADATA_PATH = "/Mounts/rbg-storage1/datasets/MIMIC/metadata_jan_2021.json"

SUMMARY_MSG = "Contructed Mimic CXR {} {} dataset with {} records, {} exams, {} patients, and the following class balance \n {}"
CXR_DATASET_NAMES = ["mimic_cxr_opacity", "mimic_cxr_atelectasis", "mimic_cxr_cardiomegaly", "mimic_cxr_consolidation", "mimic_cxr_edema",
                     "mimic_cxr_enlarged_cardiomediastinum", "mimic_cxr_fracture", "mimic_pleural_effusion", "mimic_pleural_other", "mimic_pneumonia", "mimic_pneumothorax"]


class Abstract_Mimic_Cxr(data.Dataset):
    '''MIMIC-CXR dataset
    '''

    def create_dataset(self, split_group, img_dir):
        """
        Return the dataset from the paths and labels in the json.
        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.
        """
        dataset = []
        for row in tqdm(self.metadata_json):
            pid, split, exam = row['pid'], row['split_group'], row['exam']
            split = 'dev' if row['split_group'] == 'validate' else row['split_group']
            if split != split_group:
                continue

            # TODO

            if self.check_label(row):
                label = self.get_label(row)
                dataset.append({
                    'pid': pid,
                    'path': row['path'],
                    'y': label,
                    'additional': {},
                    'exam': exam,
                })

        self.input_loader = get_sample_loader(split_group, args) # TODO

        return dataset

    def get_summary_statement(self, dataset, split_group):
        class_balance = Counter([d['y'] for d in dataset])
        exams = set([d['exam'] for d in dataset])
        patients = set([d['ssn'] for d in dataset])
        statement = SUMMARY_MSG.format(self.task, split_group, len(
            dataset), len(exams), len(patients), class_balance)
        return statement

    def get_label(self, row):
        return row['label_dict'][self.task] == "1.0"

    def check_label(self, row):
        return row['label_dict'][self.task] in ["1.0", "0.0"] or row['label_dict']['No Finding'] == "1.0"

    def __getitem__(self, index):
        sample = self.dataset[index]

        input_dict = self.get_images(sample["paths"], sample)
        sample.update(input_dict)

        return sample

    @property
    def METADATA_FILENAME(self):
        return METADATA_PATH

    # @staticmethod
    # def set_args(args):
    #    args.num_classes = 2
    #    args.num_chan = 1
    #    args.img_size = (1024, 1024)
    #    args.input_dim = args.img_size[0] * args.img_size[1]
    #    args.scramble_input_dim =  args.input_dim
    #    args.img_mean = [43.9]
    #    args.img_std = [63.2]
    #    args.input_loader_name = 'default_image_loader'
    #    args.image_augmentations = ["scale_2d", "rotate_range/min=-20/max=20"]
    #    args.tensor_augmentations = ["normalize_2d"]
    #    args.test_image_augmentations = ["scale_2d"]
    #    args.test_tensor_augmentations = ["normalize_2d"]


@register_object("mimic_pneumothorax", "dataset")
class Mimic_Cxr_Pneumothorax(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return "Pneumothorax"


@register_object("mimic_pneumonia", "dataset")
class Mimic_Cxr_Pneumonia(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return "Pneumonia"


@register_object("mimic_pleural_other", "dataset")
class Mimic_Cxr_Pleural_Other(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return "Pleural Other"


@register_object("mimic_pleural_effusion", "dataset")
class Mimic_Cxr_Pleural_Effusion(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return "Pleural Effusion"


@register_object("mimic_lung_lesion", "dataset")
class Mimic_Cxr_Lung_Lesion(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return "Lung Lesion"


@register_object("mimic_cxr_fracture", "dataset")
class Mimic_Cxr_Fracture(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return 'Fracture'


@register_object("mimic_cxr_enlarged_cardiomediastinum", "dataset")
class Mimic_Cxr_Enlarged_Cardiomediastinum(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return 'Enlarged Cardiomediastinum'


@register_object("mimic_cxr_edema", "dataset")
class Mimic_Cxr_Edema(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return 'Edema'


@register_object("mimic_cxr_consolidation", "dataset")
class Mimic_Cxr_Consolidation(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return 'Consolidation'


@register_object("mimic_cxr_cardiomegaly", "dataset")
class Mimic_Cxr_Cardiomegaly(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return 'Cardiomegaly'


@register_object("mimic_cxr_atelectasis", "dataset")
class Mimic_Cxr_Atelectasis(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return 'Atelectasis'


@register_object("mimic_cxr_opacity", "dataset")
class Mimic_Cxr_Opacity(Abstract_Mimic_Cxr):
    @property
    def task(self):
        return "Lung Opacity"
