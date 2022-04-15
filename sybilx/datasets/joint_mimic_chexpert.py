import json
from collections import Counter
from random import shuffle
from shlex import join

import torch
import numpy as np
from torch.utils import data
from tqdm import tqdm

from sybilx.utils.registry import register_object
from sybilx.utils.loading import get_sample_loader
from sybilx.datasets.nlst_xray import METAFILE_NOTFOUND_ERR

JOINT_METADATA_PATH = "/data/rsg/mammogram/CheXpert-v1.0-small/mimic_stanford_feb_2021.json"

SUMMARY_MSG = "Contructed Joint MIMIC-CXR and CheXpert {} {} dataset with {} records, {} exams, {} patients"

JOINT_TASKS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
               "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
               "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
               "Support Devices", "No Finding"]

POSSIBLE_FINDINGS = JOINT_TASKS[:-2] # "Support Devices" and "No Finding" don't count as findings

class Mimic_Chexpert_Joint_Abstract_Dataset(data.Dataset):
    '''CheXpert dataset
    '''
    def __init__(self, args, split_group):
        """
        CheXpert Dataset
        params: args - config.
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        """
        super(Mimic_Chexpert_Joint_Abstract_Dataset, self).__init__()

        self.split_group = split_group
        self.args = args

        # sanity check
        if args.dataset_file_path != JOINT_METADATA_PATH:
            print(f"WARNING! Dataset path set to unexpected path!\nexpected: {JOINT_METADATA_PATH}\nGot: {args.dataset_file_path}")

        try:
            self.metadata_json = json.load(open(args.dataset_file_path, "r"))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.dataset_file_path, e))

        self.input_loader = get_sample_loader(split_group, args)

        self.dataset = self.create_dataset(split_group)
        assert len(self.dataset) != 0

        print(self.get_summary_statement(self.dataset, split_group))

        if args.class_bal:
            assert args.num_classes == 2
            label_dist = [d[args.class_bal_key] for d in self.dataset]
            label_counts = Counter(label_dist)
            weight_per_label = 1.0 / len(label_counts)
            label_weights = {
                label: weight_per_label / count for label, count in label_counts.items()
            }

            print("Class counts are: {}".format(label_counts))
            print("Label weights are {}".format(label_weights))
            self.weights = [label_weights[d[args.class_bal_key]] for d in self.dataset]

    def create_dataset(self, split_group):
        """
        Return the dataset from the paths and labels in the json.
        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.
        """

        if self.args.assign_splits:
            np.random.seed(self.args.cross_val_seed)
            self.assign_splits(self.metadata_json)

        dataset = []
        for row in tqdm(self.metadata_json):
            pid, split, exam = row['pid'], row['split_group'], row['exam']
            split = 'dev' if row['split_group'] == 'validate' else row['split_group']
            if split != split_group:
                continue

            if row['source'] == 'mimic':
                source = 'mimic'
            else:
                assert row['source'] == 'stanford'
                source = 'chexpert'

            if self.check_label(row):
                label = self.get_label(row)
                dataset.append({
                    'pid': pid,
                    'path': row['path'],
                    'y': label,
                    'additional': {},
                    'exam': exam,
                    'source': source
                })

        return dataset

    def assign_splits(self, meta):
        assert self.args.split_type == "random"
        for idx in range(len(meta)):
            meta[idx]["split"] = np.random.choice(
                ["train", "dev", "test"], p=self.args.split_probs
            )

    def get_summary_statement(self, dataset, split_group):
        exams = set([d['exam'] for d in dataset])
        patients = set([d['pid'] for d in dataset])
        statement = SUMMARY_MSG.format(self.task, split_group, len(
            dataset), len(exams), len(patients))
        return statement

    def get_label(self, row):
        return row['label_dict'][self.task] == "1.0"

    def check_label(self, row):
        return row['label_dict'][self.task] in ["1.0", "0.0"] or row['label_dict']['No Finding'] == "1.0"

    def __getitem__(self, index):
        sample = self.dataset[index]

        input_dict = self.input_loader.get_image(sample["path"], sample)
        sample["x"] = input_dict["input"]
        #sample["mask"] = input_dict["mask"]

        return sample

    @property
    def METADATA_FILENAME(self):
        return JOINT_METADATA_PATH

@register_object("mimic+chexpert_all", "dataset")
class Mimic_Chexpert_Joint_All_Dataset(Mimic_Chexpert_Joint_Abstract_Dataset):
    def get_label(self, row):
        if self.args.treat_ambiguous_as_positive:
            y = [row['label_dict'][task] in ["1.0","-1.0"] for task in JOINT_TASKS]
        else:
            y = [row['label_dict'][task] == "1.0" for task in JOINT_TASKS]
        return torch.tensor(y)

    def check_label(self, row):
        findings_correct = all(row['label_dict'][task] in ["1.0", "0.0", "-1.0", ""] for task in JOINT_TASKS)
        any_findings = any(row['label_dict'][task] == "1.0" for task in POSSIBLE_FINDINGS)
        no_findings = row['label_dict']['No Finding'] == "1.0"
        return findings_correct and (any_findings == (not no_findings))

    @property
    def task(self):
        return "Combined"

@register_object("mimic+chexpert_pneumothorax", "dataset")
class Chexpert_Pneumothorax(Mimic_Chexpert_Joint_Abstract_Dataset):
    @property
    def task(self):
        return "Pneumothorax"


@register_object("mimic+chexpert_edema", "dataset")
class Chexpert_Edema(Mimic_Chexpert_Joint_Abstract_Dataset):
    @property
    def task(self):
        return 'Edema'


@register_object("mimic+chexpert_consolidation", "dataset")
class Chexpert_Consolidation(Mimic_Chexpert_Joint_Abstract_Dataset):
    @property
    def task(self):
        return 'Consolidation'


@register_object("mimic+chexpert_cardiomegaly", "dataset")
class Chexpert_Cardiomegaly(Mimic_Chexpert_Joint_Abstract_Dataset):
    @property
    def task(self):
        return 'Cardiomegaly'


@register_object("mimic+chexpert_atelectasis", "dataset")
class Chexpert_Atelectasis(Mimic_Chexpert_Joint_Abstract_Dataset):
    @property
    def task(self):
        return 'Atelectasis'
