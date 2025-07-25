import argparse
import json
import sys

import nltk
import numpy as np
from nltk.translate.meteor_score import meteor_score

nltk.download('wordnet', quiet=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run metrics calculation"
    )
    parser.add_argument("input_json", help="Path to input JSON file with queries.")
    args = parser.parse_args()

    # Load input JSON file
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = []
    for sample in data:
        true_ans = [sample["response"].split()]
        pred_ans = sample["pred_response"].split()

        scores.append(meteor_score(true_ans, pred_ans))

    print(f"Meteor score: {np.mean(scores)}")


if __name__ == "__main__":
    main()