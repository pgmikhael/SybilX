import pickle
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from os.path import dirname, realpath
import sys

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from sybilx.parsing import parse_args
from scripts.plcom2012.plcom2012 import PLCOm2012, PLCOresults
from sybilx.utils.registry import get_object


def main(args):
    # Load dataset and add dataset specific information to args
    print("\nLoading data...")
    test_data = get_object(args.dataset, "dataset")(args, "test")
    
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    if args.base_model == "plcom2012":
        print("-------------\nTesting on PLCOm2012")
        model = PLCOm2012(args)
        model.save_prefix = "test_plco2012"
        model.test(test_data)
    elif args.base_model == "plcoresults":
        model = PLCOresults(args)
        print("-------------\nTesting on PLCOresults")
        model.save_prefix = "test_plcoresults"
        model.test(test_data)
    else:
        raise NotImplementedError

    print("Saving args to {}".format(args.results_path))
    pickle.dump(vars(args), open("{}.args".format(args.results_path), "wb"))


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    args = parse_args()
    main(args)
