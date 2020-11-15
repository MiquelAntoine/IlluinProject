import unittest
from package.dataprocessing import (
    add_tokenized_text_column,
    add_filtered_token_column,
    add_tokens_freq_column,
)
import pandas as pd
from nltk import RegexpTokenizer


class Test(unittest.TestCase):
    def test_add_tokenized_text_column(self):
        text_test = "On ne peut pas tromper une fois mille personnes, mais on peut tromper une fois mille personnes. Euh non c'est pas ça"
        test_df = pd.DataFrame(columns=["text"])
        test_df = test_df.append([{"text": text_test}])
        tokenizer = RegexpTokenizer(r"""\w'|\w+|[^\w\s]""")
        add_tokenized_text_column(test_df, tokenizer, "text_tokens", "text")
        self.assertTrue(
            test_df.iloc[0]["text_tokens"]
            == [
                "on",
                "ne",
                "peut",
                "pas",
                "tromper",
                "une",
                "fois",
                "mille",
                "personnes",
                ",",
                "mais",
                "on",
                "peut",
                "tromper",
                "une",
                "fois",
                "mille",
                "personnes",
                ".",
                "euh",
                "non",
                "c'",
                "est",
                "pas",
                "ça",
            ]
        )

    def test_add_filtered_token_column(self):
        text_test = "On ne peut pas tromper une fois mille personnes, mais on peut tromper une fois mille personnes. Euh non c'est pas ça"
        test_df = pd.DataFrame(columns=["text"])
        test_df = test_df.append([{"text": text_test}])
        tokenizer = RegexpTokenizer(r"""\w'|\w+|[^\w\s]""")
        add_tokenized_text_column(test_df, tokenizer, "text_tokens", "text")
        add_filtered_token_column(test_df, "filtered_tokens", "text_tokens")
        self.assertTrue(
            test_df.iloc[0]["filtered_tokens"]
            == [
                "peut",
                "tromper",
                "fois",
                "mille",
                "personnes",
                "peut",
                "tromper",
                "fois",
                "mille",
                "personnes",
                "euh",
                "non",
                "ça",
            ]
        )

    def test_add_tokens_freq_column(self):
        text_test = "On ne peut pas tromper une fois mille personnes, mais on peut tromper une fois mille personnes. Euh non c'est pas ça"
        test_df = pd.DataFrame(columns=["text"])
        test_df = test_df.append([{"text": text_test}])
        tokenizer = RegexpTokenizer(r"""\w'|\w+|[^\w\s]""")
        add_tokenized_text_column(test_df, tokenizer, "text_tokens", "text")
        add_filtered_token_column(test_df, "filtered_tokens", "text_tokens")
        add_tokens_freq_column(test_df, "freq", "filtered_tokens")
        self.assertTrue(
            dict(test_df.iloc[0]["freq"])
            == {
                "peut": 2,
                "tromper": 2,
                "fois": 2,
                "mille": 2,
                "personnes": 2,
                "euh": 1,
                "non": 1,
                "ça": 1,
            }
        )
