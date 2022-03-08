import argparse
import os.path

from datasets import load_dataset, IterableDatasetDict, IterableDataset
from constants import LANGS40
from preprocess_batch import get_out_files, filter_corpus


def process_corpus(
        out_format: str,
        criteria: str,
        overwrite: bool,
        out_path: str,
        lang: str,
        num_samples: int
):
    out_files = get_out_files(out_format, overwrite, out_path, f"{lang}.txt")
    dataset: IterableDatasetDict = load_dataset("cc100", lang=("zh-Hans" if lang == "zh" else lang), streaming=True)
    got_samples = 0
    shuffled_dataset = dataset["train"].shuffle(buffer_size=10000, seed=42)
    attempts = 0
    while got_samples < num_samples:
        samples: IterableDataset = shuffled_dataset.skip(attempts * num_samples).take(num_samples)
        new_samples = filter_corpus(
            lang=lang,
            lines=samples,
            criteria=criteria,
            out_files=out_files,
            num_samples=num_samples-got_samples,
            file_mode=("w+" if attempts == 0 else "a")
        )
        got_samples += new_samples
        print(f"reached {got_samples} samples on attempt {attempts}")
        if new_samples == 0:
            return
        attempts += 1


def preprocess_for_x2static(out_format, out_path, paper, num_samples, langs=None, overwrite=False):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if langs is None:
        langs = LANGS40
    else:
        langs = langs.split()
    for lang in langs:
        process_corpus(out_format, paper, overwrite, out_path, lang, num_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", type=str, default=None,
                        help="Space-separated list of languages to do this for.")
    parser.add_argument("--criteria", type=str, default="g&j",
                        help="which criteria to apply for filtering. Currently ['g&j'|'bomm'|'mine']")
    parser.add_argument("--out_format", type=str, default="sent",
                        help="'sent' or 'para' or 'both' for whether to write out a sentence or a paragraph per line")
    parser.add_argument("--out_path", type=str, default="",
                        help="destination for processed files")
    parser.add_argument("--gpu", type=str, default="0",
                        help="index of GPU to use")
    parser.add_argument("--num_samples", type=int, default=1_000_000,
                        help="How many samples/paragraphs to take from the raw data. "
                             "Specifies max value, will use less if less data is available.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="overwrite existing output files")
    args = parser.parse_args()
    preprocess_for_x2static(args.out_format, args.out_path, args.criteria, args.num_samples, args.langs,
                            bool(args.overwrite))


if __name__ == '__main__':
    main()
