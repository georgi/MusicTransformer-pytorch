import argparse


def get_argument_parser(description=None):
    parser = argparse.ArgumentParser(description)
    parser.add_argument("-m", "--model_dir", type=str, required=True,
            help="The directory for a trained model is saved.")
    return parser