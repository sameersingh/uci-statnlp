"""Model evaluation script.

Usage:
    python evaluate.py --config config.yaml
"""
from __future__ import division
from __future__ import print_function

import argparse
import yaml


FLAGS = None


def main(_):
    # Load the configuration file.
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f)

    # Load the dataset.

    # Build the model.



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

