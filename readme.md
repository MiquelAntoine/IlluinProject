# Illuin context retrieval project

The goal of this project is to retreive the context of a question. We work with Illuin's datasets that can be found there : [fquad.illuin.tech](https://fquad.illuin.tech/).

## How to run the code

In your shell you can run this code by enter 'python main.py' with the following argurments:

Mandatory : \
--fp filepath   : the filepath of the json dataset\
--mdl modelname : the model chosen, can only be 'tf_idf'

Optional : 

--tffunc tf_function: the function used of the term frequency computation , can be 'binary' (default), 'log' or 'raw_frequency'. 

--topn n : the number of context predicted by the model (default 5)
--no_ei : disable the 'end_importance' option on the computation of the tf_idf model. This option allows a greater weight for last words of the question.

--ofn output_filename : root name for the dumps of the questions_df and contexts_df DataFrames into json. Thoses dumps will be found in the folder 'dumps'.

# Requirements

The requirement are in the requirement.txt. 
They can be simply installed with the following command" pip install -r requirements.txt"

# NLTK Data installation
But due to data needed from nltk you might need to manually install it with the following commands :\
python \
import nltk
nltk.dowload()

Then a windows should appear : tick 'all' and install. Then the code should run propely

If you encounter issues see the following tutorial [nltk.org/data.html](https://www.nltk.org/data.html)

# Some Result
The best combination of paramerters is the defaults ones ie : tf_func = 'binary'\
end_importance = True

It return an top5 accuracy of 0.78 on the train.json and 0.84 on the valid.json



