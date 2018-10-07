# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 01:18:47 2018

@author: mr_goyal
"""

#Building a chatbot with Deep NLP

#importing the libraries

import numpy as np
import tensorflow as tf
import re
import time


#####################  Part 1 - Data Preprocessing  ############################


#importing dataset

lines = open('movie_lines.txt',encoding = 'utf-8',errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt',encoding = 'utf-8',errors='ignore').read().split('\n')

#Creating Dictionary that maps each line with its id

id2line = {}
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]= _line[4]
        
    
        
#creating a list of all conversations
conversations_ids = []

for conversation in conversations[:-1]:
     _conversation  = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
     conversations_ids.append(_conversation.split(","))

#Getting Separated list of Questions and Answer

questions=[]
answers=[]

for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])


#Function to clean text
        
def  clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"won't"," will not",text)
    text = re.sub(r"can't"," cannot",text)
    text = re.sub(r"[-+=_().,\'\"{}?><%$#@!|]/","",text)
    return text

#Cleaning questions
clean_questions=[]
for question in questions:
    clean_questions.append(clean_text(question))
#Cleaning answers
clean_answers=[]
for answer in answers:
    clean_answers.append(clean_text(answer))

#Creating a dictionary which match word with its occurence
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word]= 1
        else:
            word2count[word]+=1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word]= 1
        else:
            word2count[word]+=1
