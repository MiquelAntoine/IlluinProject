# This file is about the functions used to extrat the data from a raw json
import json
import pandas as pd
from nltk import RegexpTokenizer
from nltk.corpus import stopwords


def get_dataframe_from_json_path(json_path):
    """
    Extract the data from the jsonpath and return 2 pandas DataFrame :
    -context_df : each line is a context with an id and the text of the context
    -questions_df : each line of this data is a question with an id, the text of the question and the context_id
    """
    json_file = open("train.json", encoding="utf-8")
    data = json.loads(json_file.read())["data"]

    contexts_df = pd.DataFrame(columns=["id", "text"])
    for i in range(len(data)):
        for j in range(len(data[i]["paragraphs"])):
            contexts_df = contexts_df.append(
                [
                    {
                        "id": str(i) + "_" + str(j),
                        "text": data[i]["paragraphs"][j]["context"],
                    }
                ]
            )

    questions_df = pd.DataFrame(columns=["id", "text", "context_id"])
    for i in range(len(data)):
        for j in range(len(data[i]["paragraphs"])):
            context_id = str(i) + "_" + str(j)
            for k in range(len(data[i]["paragraphs"][j]["qas"])):

                questions_df = questions_df.append(
                    [
                        {
                            "id": data[i]["paragraphs"][j]["qas"][k]["id"],
                            "text": data[i]["paragraphs"][j]["qas"][k]["question"],
                            "context_id": context_id,
                        }
                    ]
                )

    return contexts_df, questions_df


def tokenize_text_dataframe(df, tokenizer, new_column_name, target_column_name):
    """
    Add in a new column the token list of the target column get by the tokenizer
    """

    def tokenize_lower(text):
        return tokenizer.tokenize(text.lower())

    df[new_column_name] = df[target_column_name].apply(tokenize_lower)


def filter_token_dataframe(df, new_column_name, target_column_name):
    """
    Add in a new column the filtered list of tokens of the target column
    """

    french_stopwords = set(stopwords.words("french"))
    french_stopwords.update(
        [
            "quel",
            "l'",
            "c'",
            "t'",
            "d'",
            "m'",
            "n'",
            "qu'",
            "s'",
            "j'",
            ",",
            ";",
            "où",
            "?",
            "-",
            ".",
        ]
    )

    def remove_stop_words(tokens):
        return [token for token in tokens if not token in french_stopwords]

    df[new_column_name] = df[target_column_name].apply(remove_stop_words)
