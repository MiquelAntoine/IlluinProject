# This file contains all the models function that can be applied to the raw questions_df and contexts_df
from .dataprocessing import (
    get_dataframe_from_json_path,
    add_tokenized_text_column,
    add_filtered_token_column,
    add_tokens_freq_column,
)
from nltk import RegexpTokenizer
from math import log
import numpy as np


def compute_tfidf_token(token, contexts_df, save_dict, tf_function):
    # Compute the tfidf array of the token based on the contexts_df and store it in the save_dict
    list_documents_freq = list(contexts_df["context_freq"])
    list_context_id = list(contexts_df["id"])
    tf_values = np.zeros(len(list_context_id))
    nb_doc = len(list_documents_freq)
    count_occ = 0
    for i in range(len(list_context_id)):
        if token in list_documents_freq[i].keys():
            count_occ += 1

            if tf_function == "binary":
                tf_values[i] = 1

            elif tf_function == "raw_frequency" or tf_function == "log":
                tf_values[i] = list_documents_freq[i][token]

    idf_factor = log(nb_doc / (count_occ + 0.1)) / log(2)

    if tf_function == "log":
        tf_values = np.log(tf_values + 1) * idf_factor

    else:
        tf_values *= idf_factor

    save_dict[token] = tf_values


def predict_question_context_tfidf(
    question_tokens,
    contexts_df,
    tfidf_tokens_dict,
    tf_function,
    topn=1,
    end_importance=True,
):
    # Predict the context with tfidf method
    question_tfidf_array = np.zeros(len(contexts_df))

    for i in range(len(question_tokens)):
        token = question_tokens[i]
        if not token in tfidf_tokens_dict.keys():
            compute_tfidf_token(token, contexts_df, tfidf_tokens_dict, tf_function)

        tfidf_token_array = tfidf_tokens_dict[token]

        if end_importance:
            question_tfidf_array += tfidf_token_array * log(
                10 * ((i + 1) / len(question_tokens))
            )
        else:
            question_tfidf_array += tfidf_token_array

    topn_index = np.argpartition(question_tfidf_array, -topn)[-topn:]
    topn_context_ids = []
    for index in topn_index:
        topn_context_ids.append(contexts_df.iloc[index]["id"])
    return topn_context_ids


def test_tfidf(json_path, tf_function="binary", topn=1, end_importance=True):

    print("Testing tfidf method with arguments :")
    print("tf_function = ", tf_function)
    print("topn = ", topn)
    print("end_importance = ", str(end_importance))
    print("Extract raw data from", json_path)
    contexts_df, questions_df = get_dataframe_from_json_path(json_path)

    print("Preprocessing data")
    tokenizer = RegexpTokenizer(r"""\w'|\w+|[^\w\s]""")
    add_tokenized_text_column(contexts_df, tokenizer, "text_tokens", "text")
    add_tokenized_text_column(questions_df, tokenizer, "text_tokens", "text")
    add_filtered_token_column(contexts_df, "filtered_tokens", "text_tokens")
    add_filtered_token_column(questions_df, "filtered_tokens", "text_tokens")
    add_tokens_freq_column(contexts_df, "context_freq", "filtered_tokens")

    print("Preprocessing finished, starting predictions")
    # predictions
    tfidf_tokens_dict = {}
    contexts_pred = []
    for question_index in range(len(questions_df)):
        if question_index % 1000 == 0:
            print("Predicted ", question_index, "/", len(questions_df), "questions")
        question_tokens = questions_df.iloc[question_index]["filtered_tokens"]

        contexts_pred.append(
            predict_question_context_tfidf(
                question_tokens,
                contexts_df,
                tfidf_tokens_dict,
                tf_function,
                topn,
                end_importance,
            )
        )

    questions_df["contexts_pred"] = contexts_pred

    accuracy = 0
    for i in range(len(questions_df)):
        if questions_df.iloc[i]["context_id"] in questions_df.iloc[i]["contexts_pred"]:
            accuracy += 1

    print("Accurracy : ", accuracy / len(questions_df))
    return questions_df, contexts_df
