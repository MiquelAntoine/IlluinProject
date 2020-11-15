import argparse
from package import models
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp", help="filepath of the raw json")
    parser.add_argument("--mdl", help="model you wan to use")
    parser.add_argument(
        "--tffunc",
        help="tf function choosen, can only be 'log', 'binary' or 'raw_frequency'",
    )
    parser.add_argument(
        "--topn", help="The number of contexts given in prediciton", type=int
    )
    parser.add_argument("--no_ei", help="deactivate the option end_importance")

    parser.add_argument(
        "--ofn",
        help="output filename for the questions_df and the contexts_df dumps (in the folder dumps)",
    )
    args = parser.parse_args()
    json_path = args.fp
    model_name = args.mdl
    tf_function = "binary"
    if args.tffunc:
        tf_function = args.tffunc

    topn = 5
    if args.topn:
        topn = args.topn

    end_importance = True
    if args.no_ei:
        end_importance = False

    if model_name == "tf_idf":
        questions_df, contexts_df = models.test_tfidf(
            json_path, tf_function=tf_function, topn=topn, end_importance=end_importance
        )

        if args.ofn:
            output_filename = args.ofn
            (Path(__file__).parent / "dumps").mkdir(parents=True, exist_ok=True)
            questions_df.to_json(
                (Path(__file__).parent / "dumps").__str__()
                + "/"
                + output_filename
                + "_question_df.json"
            )
            contexts_df.to_json(
                (Path(__file__).parent / "dumps").__str__()
                + "/"
                + output_filename
                + "_contexts_df.json"
            )

    print("program ends")
