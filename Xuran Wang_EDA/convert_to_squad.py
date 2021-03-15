import argparse
import json
import logging
import os
import re
import gzip
import string

from fuzzywuzzy import fuzz
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main(input_path, output_path, verbose):
    squad_data = {"data": [], "version": "1.1"}

    logger.info(f"Loading data from {input_path}")
    num_questions = 0.0

    if os.path.isfile(input_path):
        squad_data["data"].extend(read_file(input_path, verbose))

    # Count number of questions
    for example in squad_data["data"]:
        for paragraph in example["paragraphs"]:
            for question_answer in paragraph["qas"]:
                num_questions += 1

    # Verify the data
    assert "data" in squad_data.keys()
    assert "version" in squad_data.keys()
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qas in paragraph["qas"]:
                assert qas["question"]
                for answer in qas["answers"]:
                    assert (
                        answer["text"]
                        == context[
                            answer["answer_start"] : answer["answer_start"] + len(answer["text"])
                        ]
                    )

    logger.info(f"Writing output to {output_path}")
    logger.info(f"Number of questions: {num_questions}")
    with open(output_path, "w") as output_file:
        json.dump(squad_data, output_file)


def read_file(input_path, verbose):
    instances = []
    with gzip.open(input_path,'rb') as input_file:
        for line in tqdm(input_file, leave=False):
            mrqa_instance = json.loads(line)
            if "header" in mrqa_instance:
                continue
            passage = mrqa_instance["context"]
            passage = passage.replace("\xa0", " ").replace("\u2019", "'")
            questions_detected_answers = [
                qa.get("detected_answers", []) for qa in mrqa_instance["qas"]
            ]
            questions_allowed_answers = [qa.get("answers", []) for qa in mrqa_instance["qas"]]
            if not all([len(x) != 0 for x in questions_detected_answers]):
                raise ValueError(f"Instance has question with no detected answers: {mrqa_instance}")
            questions_squad_format_answers = []
            for question_allowed_answers, question_detected_answers in zip(
                questions_allowed_answers, questions_detected_answers
            ):
                question_squad_format_answers = []
                for detected_answer in question_detected_answers:
                    # Special-case the scenario where there is literally only 1 option
                    # (one character span). We _need_ to use that one, regardless of
                    # how it matches with the text.
                    if len(detected_answer["char_spans"]) == 1:
                        start_char_span = detected_answer["char_spans"][0][0]
                        end_char_span = detected_answer["char_spans"][0][1] + 1
                        question_squad_format_answers.append(
                            {
                                "answer_start": start_char_span,
                                "text": passage[start_char_span:end_char_span],
                            }
                        )
                        # Move on to the other detected answers
                        continue
                    detected_answer_text = detected_answer["text"]
                    # Sometimes, the detected answer was found using heuristics. So, what we want to do
                    # is to take the start char span according to the detected answer, and then get the next
                    # character for the length of the detected answer text. This is basically truecasing
                    # the detected answer with the passage information.
                    matching_answer_strings = []
                    matching_start_char_spans = []

                    # Also allow matches with normalized answers, with curly quotes
                    # converted to straight quotes
                    question_allowed_answers += [
                        straighten_curly_quotes(answer) for answer in question_allowed_answers
                    ]
                    normalized_allowed_answers = [
                        squad_normalize_answer(ans) for ans in question_allowed_answers
                    ]
                    # Also allow matches with normalized answers, minus spaces.
                    normalized_allowed_answers_nospace = [
                        re.sub(r"\s+", "", ans, flags=re.UNICODE)
                        for ans in normalized_allowed_answers
                    ]
                    normalized_allowed_answers = (
                        normalized_allowed_answers + normalized_allowed_answers_nospace
                    )
                    for char_span in detected_answer["char_spans"]:
                        start_char_span = char_span[0]
                        end_char_span = char_span[1]
                        matched_answer_text_from_passage = passage[
                            start_char_span : start_char_span + len(detected_answer_text)
                        ]
                        answer_text_from_detected_answer_spans = passage[
                            start_char_span : end_char_span + 1
                        ]
                        # Check if squad normalized found answer text is in the normalized
                        # allowed answers. MRQA also does some weird stuff with hyphens
                        # and apostrophes
                        # (e.g., turning them into a space instead of removing them),
                        # so we check that as well.
                        if (
                            (
                                squad_normalize_answer(matched_answer_text_from_passage)
                                in normalized_allowed_answers
                            )
                            or (
                                squad_normalize_answer(
                                    matched_answer_text_from_passage.replace("-", " ")
                                )
                                in normalized_allowed_answers
                            )
                            or (
                                squad_normalize_answer(
                                    matched_answer_text_from_passage.replace("'", " ")
                                )
                                in normalized_allowed_answers
                            )
                        ):
                            matching_answer_strings.append(matched_answer_text_from_passage)
                            matching_start_char_spans.append(start_char_span)
                        if (
                            (
                                squad_normalize_answer(answer_text_from_detected_answer_spans)
                                in normalized_allowed_answers
                            )
                            or (
                                squad_normalize_answer(
                                    answer_text_from_detected_answer_spans.replace("-", " ")
                                )
                                in normalized_allowed_answers
                            )
                            or (
                                squad_normalize_answer(
                                    answer_text_from_detected_answer_spans.replace("'", " ")
                                )
                                in normalized_allowed_answers
                            )
                        ):
                            matching_answer_strings.append(answer_text_from_detected_answer_spans)
                            matching_start_char_spans.append(start_char_span)
                    if matching_answer_strings:
                        # Break ties between the matching answer strings by maximizing the
                        # levenshtein ratio with any of the allowed answers.
                        best_ratio = -1.0
                        best_answer_string = None
                        best_start_char_span = None
                        # Maintain a set of checked answers, so we can skip duplicates
                        checked_answers = set()
                        for answer, start_span in zip(
                            matching_answer_strings, matching_start_char_spans
                        ):
                            if answer in checked_answers:
                                continue
                            for question_allowed_answer in question_allowed_answers:
                                ratio = fuzz.ratio(question_allowed_answer.lower(), answer.lower())
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_answer_string = answer
                                    best_start_char_span = start_span
                            checked_answers.add(answer)
                        question_squad_format_answers.append(
                            {"answer_start": best_start_char_span, "text": best_answer_string}
                        )
                    else:
                        print(
                            "WARNING: Couldn't get any of the detected answers "
                            "to align exactly with a span in the passage."
                        )
                        print(f"passage: {passage}")
                        print(f"detected answers: {question_detected_answers}")
                questions_squad_format_answers.append(question_squad_format_answers)
            squad_format_qas = []
            # Only keep questions where we have answers.
            questions = [
                qa["question"]
                for (qa, ans) in zip(mrqa_instance["qas"], questions_squad_format_answers)
                if ans
            ]
            qids = [
                qa["qid"]
                for (qa, ans) in zip(mrqa_instance["qas"], questions_squad_format_answers)
                if ans
            ]
            questions_squad_format_answers = [ans for ans in questions_squad_format_answers if ans]
            assert len(questions) == len(qids)
            assert len(qids) == len(questions_squad_format_answers)
            for question, qid, squad_format_answers in zip(
                questions, qids, questions_squad_format_answers
            ):
                question = question.replace("\xa0", " ").replace("\u2019", "'")
                squad_format_qas.append(
                    {"question": question, "id": qid, "answers": squad_format_answers}
                )
            new_instance = {
                "title": passage[:50],
                "paragraphs": [{"context": passage, "qas": squad_format_qas}],
            }
            instances.append(new_instance)
    return instances


def squad_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def straighten_curly_quotes(text):
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s " "- %(name)s - %(message)s", level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description=("Convert a MRQA-formatted dataset into SQuADv1.1 format."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-path", type=str, required=True, help=("Path to MRQA-format data to convert.")
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help=("Path prefix to write SQuADv1.1-formatted dataset."),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print warnings when detected answer in MRQA instance "
            "doesn't match paragraph-extracted answer."
        ),
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.verbose)
