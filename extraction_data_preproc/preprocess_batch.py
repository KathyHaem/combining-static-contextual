import argparse
import os
from typing import List, Union, Dict, Any

import fugashi
import jieba
import spacy_udpipe
from datasets import IterableDataset
from icu_tokenizer import SentSplitter
from icu_tokenizer import Tokenizer
from tqdm import tqdm

from constants import LANGS_IN_UDPIPE


def keep(segmented: List[List[str]], criteria="g&j") -> bool:
    """ filtering logic as per gupta & jaggi """
    if criteria == "g&j":
        if len(segmented) < 3:
            return False
        char_len = sum([sum([len(tok) for tok in sent]) for sent in segmented])
        return char_len >= 140  # may not be suitable for all languages!
    elif criteria == "mine":
        return len(segmented) >= 2  # i.e., avoid headlines
    else:
        return True


def write_segment(segmented: List[List[str]], fout, out_format: str, criteria: str):
    """ actual writing process depending on format. """
    written = 0
    if out_format == "sent":
        for sent in segmented:
            if criteria == "bomm" and (len(sent) < 7 or len(sent) > 75):
                continue
            if criteria == "mine" and (len(sent) < 7):  # avoid really short sentences
                continue
            out = " ".join(sent)
            fout.write(out + "\n")
            written += 1
        return written
    elif out_format == "para":
        out = " ".join([" ".join(sent) for sent in segmented])
        fout.write(out + "\n")
        return 1


def filter_corpus(
        lang: str,
        lines: Union[List[str], IterableDataset],
        out_files: Dict[str, Any],
        criteria: str,
        num_samples: int,
        file_mode: str = "w+"
):
    """ tokenise, filter and write out required formats """
    if not out_files:
        print(f"out_files for {lang} is empty. you may need to specify --overwrite or a correct --out_format")
        return

    annotated = segment_tokenize(lang, lines, num_samples)

    for out_format in out_files:
        out_files[out_format] = open(out_files[out_format], file_mode, encoding="utf-8")

    written = 0
    for ann in tqdm(annotated, desc="writing out tokenized"):
        if written >= num_samples:
            print(f"reached num_samples ({num_samples})")
            break
        segmented = []
        if isinstance(ann, list):
            segmented = ann
        else:
            for sent in ann.sentences:
                tokens = [token.text.lower() for token in sent.tokens]
                segmented.append(tokens)

        if keep(segmented, criteria):
            sents_written = 0
            for out_format in out_files:
                sents_written = write_segment(segmented, out_files[out_format], out_format, criteria)
            written += sents_written

    for out_format in out_files:
        name = out_files[out_format].name
        out_files[out_format].close()
        out_files[out_format] = name
    return written


def segment_tokenize(
        lang: str,
        lines: Union[List[str], IterableDataset],
        num_samples: int
) -> List[List[List[str]]]:
    """ actual tokenisation, depending on lang """
    is_dataset = isinstance(lines, IterableDataset)
    lines = [line["text"] if is_dataset else line for line in lines]

    if lang not in LANGS_IN_UDPIPE:
        splitter = SentSplitter(lang=lang)
        tokenizer = Tokenizer(lang=lang)
        if lang == "ja":
            mecab = fugashi.Tagger()
        result = []
        for line in tqdm(lines, desc="segmenting and tokenizing", total=num_samples):
            segmented = []
            split = splitter.split(line)
            for sent in split:
                if lang == "ja":
                    tokens = [word.surface for word in mecab(sent)]
                elif lang == "zh":
                    tokens = list(jieba.cut(sent, cut_all=False))
                else:
                    tokens = tokenizer.tokenize(sent)
                segmented.append(tokens)
            result.append(segmented)
        return result

    else:
        spacy_udpipe.download(lang)
        nlp = spacy_udpipe.load(lang)
        udpipe_result = nlp.pipe(lines, n_process=16)
        result = []
        try:
            for doc in tqdm(udpipe_result, desc="segmenting and tokenizing with udpipe", total=num_samples):
                segmented = []
                for sent in doc.sents:
                    tokens = [t.text for t in sent]
                    segmented.append(tokens)
                result.append(segmented)
        except ValueError as e:
            print(e)
            print("exiting segment_tokenize but will try again")
    return result


def add_out_format(
        out_path: str, overwrite: bool, out_format: str, corpus: str, out_files: Dict[str, Any]
) -> Dict[str, Any]:
    """ paragraph, sentence, or both? """
    filtered = os.path.join(out_path, f"{out_format}_{corpus}")
    print(f"Will write {out_format} data to file {filtered}")
    if not os.path.exists(filtered) or overwrite:
        out_files[out_format] = filtered
    return out_files


def process_corpus(args, corpus: str, lang: str):
    """ wrapper function for per-language tasks """
    print("========== {} ===========".format(corpus))
    out_files = get_out_files(args.out_format, args.overwrite, args.out_path, corpus)
    with open(os.path.join(args.in_path, corpus), "r", encoding="utf-8") as fin:
        print("reading file..")
        lines = fin.readlines()
    filter_corpus(lang, lines, out_files, args.paper, args.num_samples)


def get_out_files(out_format: str, overwrite: bool, out_path: str, corpus: str) -> Dict[str, Any]:
    out_files: dict = {}
    if out_format == "sent" or out_format == "both":
        out_files = add_out_format(out_path, overwrite, "sent", corpus, out_files)
    if out_format == "para" or out_format == "both":
        out_files = add_out_format(out_path, overwrite, "para", corpus, out_files)
    return out_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="",
                        help="Folder where source corpora are..")
    parser.add_argument("--out_path", type=str, default="",
                        help="Folder to write everything out to.")
    parser.add_argument("--out_format", type=str, default="both",
                        help="'sent' or 'para' or 'both' for whether to write out a sentence or a paragraph per line")
    parser.add_argument("--langs", type=str, default=None,
                        help="Space-separated list of languages to do this for. If None, will do os.listdir instead")
    parser.add_argument("--paper", type=str, default="g&j",
                        help="which criteria to apply for filtering. Currently ['g&j'|'bomm']")
    parser.add_argument("--gpu", type=str, default="0",
                        help="index of GPU to use (will set CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--num_samples", type=int, default=100_000,
                        help="How many samples/paragraphs to take from the raw data. "
                             "Specifies max value, will use less if less data is available.")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size for stanza batch processing")
    parser.add_argument("--overwrite", type=bool, default=False,
                        help="overwrite existing output files")
    args = parser.parse_args()

    # set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ensure out_dir exists
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    if not args.langs:
        for corpus in os.listdir(args.in_path):
            process_corpus(args, corpus, os.path.splitext(corpus)[0])
    else:
        for lang in args.langs.split():
            process_corpus(args, f"{lang}.txt", lang)


if __name__ == "__main__":
    main()
