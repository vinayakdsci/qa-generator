
import transformers
from transformers import (
    # Text2TextGenerationPipeline,
    AutoModelForSeq2SeqLM as alwm,
    # TokenClassificationPipeline,
    # AutoModelForTokenClassification,
    AutoModelForQuestionAnswering as amqa,
    AutoTokenizer as att,
    # BertTokenizer,
    # AlbertTokenizer,
    # BertForQuestionAnswering,
    # AlbertForQuestionAnswering,
    # T5Config,
    # T5ForConditionalGeneration, 
    T5TokenizerFast,
    PreTrainedTokenizer,
    PreTrainedModel,
    # ElectraTokenizer,
    # ElectraForQuestionAnswering
)
import torch
import string
import numpy as np
from transformers import pipeline
from transformers.pipelines import AggregationStrategy
import pickle

# sq_tokenizer = att.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
# sq_model = alwm.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
# text= "The abolition of feudal privileges by the National Constituent Assembly on 4 August 1789 and the Declaration \\nof the Rights of Man and of the Citizen (La Déclaration des Droits de l'Homme et du Citoyen), drafted by Lafayette \\nwith the help of Thomas Jefferson and adopted on 26 August, paved the way to a Constitutional Monarchy \\n(4 September 1791 – 21 September 1792). Despite these dramatic changes, life at the court continued, while the situation \\nin Paris was becoming critical because of bread shortages in September. On 5 October 1789, a crowd from Paris descended upon Versailles \\nand forced the royal family to move to the Tuileries Palace in Paris, where they lived under a form of house arrest under \\nthe watch of Lafayette's Garde Nationale, while the Comte de Provence and his wife were allowed to reside in the \\nPetit Luxembourg, where they remained until they went into exile on 20 June 1791."
hftokenizer = pickle.load(open('models/hftokenizer.sav', 'rb'))
hfmodel = pickle.load(open('models/hfmodel.sav', 'rb'))

def run_model(input_string, **generator_args):
  generator_args = {
  "max_length": 256,
  "num_beams": 4,
  "length_penalty": 1.5,
  "no_repeat_ngram_size": 3,
  "early_stopping": True,
  }
  # tokenizer = att.from_pretrained("ThomasSimonini/t5-end2end-question-generation")
  input_string = "generate questions: " + input_string + " </s>"
  input_ids = hftokenizer.encode(input_string, return_tensors="pt")
  res = hfmodel.generate(input_ids, **generator_args)
  output = hftokenizer.batch_decode(res, skip_special_tokens=True)
  output = [item.split("<sep>") for item in output]
  return output
  
al_model = pickle.load(open('models/al_model.sav', 'rb'))
al_tokenizer = pickle.load(open('models/al_tokenizer.sav', 'rb'))
def QA(question, context):
  # model_name="deepset/electra-base-squad2"
  nlp = pipeline("question-answering",model=al_model,tokenizer=al_tokenizer)
  format = {
      'question':question,
      'context':context
  }
  res = nlp(format)
  output = f"{question}\n{string.capwords(res['answer'])}\tscore : [{res['score']}] \n"
  return output
  # inputs = tokenizer(question, context, return_tensors="pt")
  # # Run the model, the deepset way
  # with torch.no_grad():
  #   output = model(**inputs)
  # start_score = output.start_logits
  # end_score = output.end_logits
  # #Get the rel scores for the context, and calculate the most probable begginign using torch
  # start = torch.argmax(start_score)
  # end = torch.argmax(end_score)
  # #cinvert tokens to strings
  # # output = tokenizer.decode(input_ids[start:end+1], skip_special_tokens=True)
  # predict_answer_tokens = inputs.input_ids[0, start : end + 1]
  # output = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
  # output = string.capwords(output)
  # print(f"Q. {question} \n Ans. {output}")
# QA("What was the first C program","The first prgram written in C was Hello World")

def gen_question(inputs):

 questions = run_model(inputs)

 return questions

# string_query = "Hello World"
# gen_question(f"answer: {string_query} context: The first C program said {string_query} ").  #The format of the query to generate questions


def read_file(filepath_name):
  with open(text, "r") as infile:
    contents = infile.read()
    context = contents.replace("\n", " ")
  return context

def create_string_for_generator(context):
    gen_list = gen_question(context)
    return (gen_list[0][0]).split('? ')

def creator(context):
  questions = create_string_for_generator(context)
  pairs = []
  for ques in questions:
    pair = QA(ques,context)
    pairs.append(pair)
  return pairs

# sentences = main_text.split('.')
  # creator(sent)
