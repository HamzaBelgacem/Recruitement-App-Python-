import csv
import pandas as pd
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import docx2txt
import PyPDF2
import os
from os import listdir
from os.path import isfile, join
from io import StringIO
import pandas as pd
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher
# with open('names.csv', 'w', newline='') as csvfile:
#     fieldnames = ['first_name', 'last_name','hello','hhhhh','pppppp']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
#     writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
#     writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})

keyword_dict = pd.read_csv('names.csv',sep=",")
print(keyword_dict)
# df = pd.read_csv('template_new.csv',sep=";")
# print(df)
stats_words = [nlp(text) for text in keyword_dict['first_name'].dropna(axis = 0)]
NLP_words = [nlp(text) for text in keyword_dict['last_name'].dropna(axis = 0)]
ML_words = [nlp(text) for text in keyword_dict['hello'].dropna(axis = 0)]
DL_words = [nlp(text) for text in keyword_dict['hhhhh'].dropna(axis = 0)]
R_words = [nlp(text) for text in keyword_dict['pppppp'].dropna(axis = 0)]
print("okkkkkk")