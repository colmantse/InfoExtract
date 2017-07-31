import pickle
'''
vocab=pickle.load(open('data/vocab.pak','rb'))

from util import load_data
from bpe import wordpiece
from collections import Counter

# create a dict obj for training a bpe model
tokenized_corpus=load_data('data/all_cases.data',mode='token')
vocab = Counter(tokenized_corpus)

#train a bpe model until converge, it has the highest priority
bpe_model=wordpiece(vocab,until_converge=True)

#train a bpe model by iteration, note it does not reflect the number of wordpiece 
#since a group of them are obtained each iteration 
bpe_model=wordpiece(vocab,iteration=10000)

#train a bpe model by the number of units, note it will stop as soon as the wp units 
#obtained at the end of the iteration exceed the specification, priority of unit override iteration
bpe_model=wordpiece(vocab,units=10000)

'''

'''
#import pickle
import time
from util import read_large_data
from model_word2vec import word2vec_model
#bpe_model=pickle.load(open('data/model/bpe_14760_pieces.wp','rb'))
words=read_large_data('data/all_cases_cleaned.data')

#The following shows how to use the word2vec_model to train embedding, cbow or skipgram.

t1=time.time()

#model=word2vec_model(bpe_model,True) #optionally use bpe
model=word2vec_model()
model.build_dataset(words,min_count=500)
model.init_train(batch_size=21,use_skipgram=False) #this is cbow, if use skipgram, set to True
model.view_valid()
model.train(len(words)//21) #number of steps

t2=time.time()
print('profile:',t2-t1)

model.plot_embedding()
'''
import pickle,os
from main import load_all, load
from model_word2vec import word2vec_model

#bpe_model=pickle.load(open('data/model/bpe_14760_pieces.wp','rb'))

model=pickle.load(open('data/model/w2v_model_cleaned','rb'))
model.name='w2v_model_cleaned'
#model.wp2idx,model.idx2wp,model.final_embedding=pickle.load(open('data/all_cases_wp.weights','rb'))

from data_builder import data_builder
data_model=data_builder(model,use_model=True)

'''
_courts=[x for x in os.listdir() if '.txt' in x and x[0]=='_']
show=[x for x in os.listdir() if '.txt' in x and x not in _courts]

#this use court cases as data
a=load(show[0])
data_model.build_data_from_court_cases(load(show[-1]),a,'interpreter')
del a

'''
#this use monolingual text as data (preprocessed)
with open('data/europarl-v7.en','r',encoding='utf8') as f:
  text=f.read()

size=len(text)//10
text=text[:size]
data_model.build_data_from_text(text)

data_model.plot_sequence_length()  

train_ratio=0.66
inc_bg_ratio=0.02
num_gen_data=5
max_length=27
min_length=0

for i in range(num_gen_data):
  data_path='data/gen_data/{}_{}.gendata'.format(data_model.name,i)
  pickle.dump(data_model.generate_data(train_ratio,inc_bg_ratio,max_length,min_length),open(data_path,'wb'),protocol=pickle.HIGHEST_PROTOCOL)

data_model_name=data_model.name
del data_model

from rnn_classifier import classifier

class struct:
  def __init__(self,use_attention_decoder,use_bidirection,use_decoder,classify_seq,seq2seq,seq_filter_weight,number_of_layers,num_hidden,max_length,drop_out):
  
    assert not (classify_seq is True and classify_seq is seq2seq)
    assert number_of_layers>0
    assert seq_filter_weight>=0 and seq_filter_weight<=1
    assert num_hidden>0

    self.use_attention_decoder=use_attention_decoder
    self.use_bidirection=use_bidirection
    self.use_decoder=use_decoder
    self.classify_seq=classify_seq
    self.seq2seq=seq2seq
    self.seq_filter_weight=seq_filter_weight
    self.number_of_layers=number_of_layers
    self.num_hidden=num_hidden
    self.max_length=max_length
    self.drop_out=drop_out

    self.name=''
    if use_attention_decoder:
      self.name+='_attention_'
    if use_bidirection and not use_attention_decoder:
      self.name+='_bidirectional_'
    if use_decoder and not use_attention_decoder:
      self.name+='_decoder_'
    else:
      self.name+='_encoder_'
    if classify_seq:
      self.name+='_classifier_'
    if seq2seq:
      self.name+='_seq2seq_'
    if not classify_seq and not seq2seq:
      self.name+='_filter_0{}'.format(int(seq_filter_weight*100))

'''
TO EVAL:
1. Attention sequence filter-classifier ratio 0.25:    struct(True,False,False,False,False,0.25,3,256,27,0.6)
2. Attention sequence filter-classifier ratio 0.75:    struct(True,False,False,False,False,0.75,3,256,27,0.6)
3. Attention sequence classifier:                      struct(True,False,False,True,False,0,3,256,27,0.6)
4. Bidirectional decoder filter-classifier ratio 0.25: struct(False,True,True,False,False,0.25,3,256,27,0.6)
5. Bidirectional decoder filter-classifier ratio 0.75: struct(False,True,True,False,False,0.75,3,256,27,0.6)
6. Bidirectional decoder classifier:                   struct(False,True,True,True,False,0,3,256,27,0.6)
'''

struct_to_eval=[struct(True,False,False,False,False,0.25,3,256,27,0.6),
                struct(True,False,False,False,False,0.75,3,256,27,0.6),
                struct(True,False,False,True,False,0,3,256,27,0.6),
                struct(False,True,True,False,False,0.25,3,256,27,0.6),
                struct(False,True,True,False,False,0.75,3,256,27,0.6),
                struct(False,True,True,True,False,0,3,256,27,0.6)]

data_list=[x for x in os.listdir('data/gen_data/') if data_model_name in x]
  
for data_name in data_list:
  for _struct in struct_to_eval:
    c_model=classifier(model,use_model=True)
    c_model.create_structure(use_attention_decoder=_struct.use_attention_decoder,
                               use_bidirection=_struct.use_bidirection,
                               use_decoder=_struct.use_decoder,
                               classify_seq=_struct.classify_seq,
                               seq2seq=_struct.seq2seq,
                               seq_filter_weight=_struct.seq_filter_weight,
                               number_of_layers=_struct.number_of_layers,
                               num_hidden=_struct.num_hidden,
                               max_length=_struct.max_length,
                               drop_out=_struct.drop_out)
  
    graph_name='logs/{}_{}/'.format(data_name,_struct.name)
    d_name='data/gen_data/{}'.format(data_name)
    c_model.train(batch_size=10,epoch=7,record_interval=1,graph_directory=graph_name,data_path=d_name)
    del c_model