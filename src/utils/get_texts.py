import argparse
import json
import string
from pathlib import Path

# def preprocess_text(text):
#     if text.count(".") == 1 and text[-1] == ".":
#         return text
#     print(text)
#     sents = text.split(".")
#     sents = [s for s in sents if len(s) > 0]
#     print("->\t" + str(sents))
#     result = []
#     for s in sents:
#         if s[-1] in string.punctuation:
#             s = s[:-1] + "."
#         result.append(s)
#         print("->\t\t" + s)

#     return result


def preprocess_text(text):
    if text[-1] in string.punctuation:
        text = text[:-1] + "."
    else:
        print(text)
        text = text + "."

    return [text]


def extract_texts(json_path, output_path):
    texts = []

    with open(json_path, "r") as in_file:
        for line in in_file.readlines():
            line = line.strip()
            d = json.loads(line)
            texts.extend(preprocess_text(d["text"]))

    with open(output_path, "w") as out_file:
        for i, text in enumerate(texts):
            out_file.write(text)
            if i < len(texts) - 1:
                out_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    extract_texts(args.json_path, args.output_path)
