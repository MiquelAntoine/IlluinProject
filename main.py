import argparse
import models

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
        models.test_tfidf(
            json_path, tf_function=tf_function, topn=topn, end_importance=end_importance
        )

    print("progam ends")
