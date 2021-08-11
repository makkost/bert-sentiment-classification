import argparse

BASE_MODEL_DIR = "rubert_cased_L-12_H-768_A-12_v2"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the data.", nargs="+")
    parser.add_argument("--split", help="Number of datasets for cross-validation.", default=4, type=int)
    parser.add_argument("--model", help="Path to the model.", default=BASE_MODEL_DIR)
    parser.add_argument("--output", help="Directory where datasets will be stores.", default="data/")
    return parser.parse_args()