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

#Creating 2 dictionary that map the question and answer rto unique integer
threshold = 20
questionswords2int = {}
answerswords2int = {}
word_number = 0
for word,count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        answerswords2int[word] = word_number
        word_number += 1

#Adding the last  2 tokens for these two dictionary
        
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']

for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
    answerswords2int[token] = len(answerswords2int) + 1
    
#Adding EOS token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
    
# Translating all the list of question and answer to integers and replacing all the words that are 
# filter out by <OUT>
questions_into_int = []

for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)

answers_into_int = []

for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)


#Sorted questions and answers by length of questions

sorted_clean_questions= []
sorted_clean_answers= []

for length in range(1,25+1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])


#Creating Placeholder for input and target

def model_input():
    inputs = tf.placeholder(tf.int32,[None,None],name='input')
    target = tf.placeholder(tf.int32,[None,None],name='target')
    lr = tf.placeholder(tf.float32,name='learning_rate')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    return inputs,target,lr,keep_prob

#Preprocessing target
def preprocess_targets(targets,word2int,batch_size):
    left_side = tf.fill([batch_size,1],word2int['<SOS>'])
    right_side = tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])
    preprocessed_targets = tf.concat([left_side,right_side],1)
    return preprocessed_targets

#Encoder RNN Layer
def encoder_rnn_layer(rnn_inputs,rnn_size,num_layers,keep_prob,sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    _,encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                      cell_bw = encoder_cell,
                                                      sequence_length = sequence_length,
                                                      dtype = tf.float32,
                                                      inputs = rnn_inputs)
    return encoder_state
