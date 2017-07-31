import random,collections
import gensim,logging,numpy as np
from gensim import matutils
from numpy import array, float32 as REAL
from util import clean_text,clean,randomize_sequence,valid_weight
from gensim.summarization import textcleaner
import matplotlib.pyplot as plt

#For using gensim model
def hash32(value):
     return hash(value) & 0xffffffff

class data_builder:

  def __init__(self,model=None, use_model=False):
    self.model=model
    self.use_model=use_model
    self.data_is_text=False
    
    if use_model is True:
      assert model is not None

  def build_data_from_text(self,text,ratio=3,use_bpe=False):
    '''This function output valid and random generated invalid sentences.
       This method assume text to be sentences e.g. europarl corpus
    '''
    self.data_is_text=True
    self.name='{}'.format(self.model.name)
    text=text.split('\n')
    non_text=[[0]]*(len(text)*ratio) #allocate memory
   
    if use_bpe:
      assert self.model is not None and self.model.bpe_model is not None
      for i,s in enumerate(text):
        text[i]=self.model.bpe_model.wp_sent(s).split(' ')
        while '' in text[i]:
          text[i].remove('')
        for j in range(ratio):
          rs=randomize_sequence(s)
          non_text[i*ratio+j]=rs if rs is None else (rs[0],rs[1],1)
        text[i]=(text[i],[-1,0,len(text[i])],0) #-1 in the front signifies valid sentence      
 
    if self.use_model or use_bpe and not self.use_model:
      for i,s in enumerate(text):
        s=s.split(' ')
        while '' in s:
          s.remove('')
        text[i]=self.model.numerize_data(s)
        for j in range(ratio):
          rs=randomize_sequence(text[i])
          non_text[i*ratio+j]=rs if rs is None else (rs[0],rs[1],1)
        text[i]=(text[i],[-1,0,len(text[i])],0) #-1 in the front signifies valid sentence	
   
    else: #not use model and not use bpe, build vocab list
      self.no_model_wp2idx={}
      for i,sent in enumerate(text):
        text[i]=clean(sent).split(' ')
        while '' in text[i]:
          text[i].remove('')
        #may not be faster than writing the whole thing down and use gensim
        for w in text[i]:
          if w in self.no_model_wp2idx:
            self.no_model_wp2idx[w]=len(self.no_model_wp2idx)+1 #0 is reserved for padding

      self.no_model_idx2wp = dict(zip(self.no_model_wp2idx.values(), self.no_model_wp2idx.keys()))

      for i,s in enumerate(text):
        for j,w in enumerate(s):
          text[i][j]=self.no_model_wp2idx[w]
        for j in range(ratio):
          rs=randomize_sequence(s)
          non_text[i*ratio+j]=rs if rs is None else (rs[0],rs[1],1)
        text[i]=(text[i],[-1,0,len(text[i])],0) #-1 in the front signifies valid sentence      

      '''
      #use gensim
      path='data/tmp/tmp.data'      
      text_len=len(text) #going to delete text to save memory

      with open(path,'w',encoding='utf8') as f:
        while len(text)!=0:
          f.write(clean(text[0])+' ')
          del text[0] #save memory

      logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
      model=gensim.models.Word2Vec(workers=3,sg=0,hs=0,negative=5,cbow_mean=1,size=300, hashfxn=hash32) 
      sentences=gensim.models.word2vec.LineSentence(path)
      model.build_vocab(sentences)

      self.no_model_wp2idx={}
      for wp in model.word2index:
        self.no_model_wp2idx[wp]=len(self.no_model_wp2idx)+1 #0 is reserved for padding
      self.no_model_idx2wp = dict(zip(self.no_model_wp2idx.values(), self.no_model_wp2idx.keys()))

      text=[[0]]*text_len #allocate memory
      i=0
      with open(path,'r',encoding='utf8') as f:
        t=f.readline()
        t=t.split(' ')
   
        #turn each token in sentence in text to id
        for j,w in enumerate(t):
          t[j]=self.no_model_wp2idx[w]
        text[i]=t
        for j in range(ratio):
          rs=randomize_sequence(s)
          non_text[i*ratio+j]=rs if rs is None else (rs[0],rs[1],1)
        text[i]=(s,[-1,0,len(s)],0) #-1 in the front signifies valid sentence      

      '''

    self.rev_data=text
    self.unrev_data=non_text
    self.bg_data=None
    	
  def build_data_from_court_cases(self,rev_cases,unrev_cases,keyword, use_bpe=False, inc_bg=True):
    '''This function reads a keyword and output relevant and unrelevant sentences
       This function assumes a supervised scenario where a dataset of relevant and 
       unrelevant cases are provided.'''
  
    if use_bpe:
      assert self.model is not None and self.model.bpe_model is not None 

    self.name='{}_court_keyword_{}'.format(self.model.name,keyword)

    #obtain the sentences
    rev_cases._clean_sentences()
    unrev_cases._clean_sentences()

    rev_sents,unrev_sents=[],[]
    bg_sents=[] if inc_bg else None

    #The followign codes hope to save effort in: 
    #1. extending rev_cases.sents and unrev_cases.sents
    #2. one additional iteration over rev_sents and unrev_sents for bg_sents
    #However, there might be additional overhead for append

    for i,sents in enumerate(rev_cases.sents):
      for sent in sents:
        if keyword in sent:
          rev_sents.append(sent)
        else:
          if inc_bg:
            bg_sents.append(sent)

    #we did not put rev_cases.sents and unrev_cases.sents into the same list
    #because of additional time spent on extension and memory consideration.

    for i,sents in enumerate(unrev_cases.sents):
      for sent in sents:
        if keyword in sent:
          unrev_sents.append(sent)
        else:
          if inc_bg:
            bg_sents.append(sent)

    print('rev data: {} unrev data: {}'.format(len(rev_sents),len(unrev_sents)))

    def find_keyword_number(seq,k_seq):
      #k_seq must be a list of tokens(id)
      assert isinstance(k_seq,list)

      key_pos=[]
      pointer=0
      while pointer<len(seq):
        key_pointer=0
        hold_pointer=pointer

        while pointer<len(seq) and key_pointer<len(k_seq) and seq[pointer]==k_seq[key_pointer]:
          key_pointer+=1
          if key_pointer==len(k_seq):
            key_pos.extend([i for i in range(hold_pointer,pointer)])
          else:
            pointer+=1
        pointer+=1

      return key_pos

    if use_bpe:
      keyword=self.model.bpe_model.wp_sent(keyword).split(' ')
      while '' in keyword:
        keyword.remove('')

      for i,s in enumerate(rev_sents):
        rev_sents[i]=self.model.bpe_model.wp_sent(s).split(' ')
        while '' in rev_sents[i]:
          rev_sents[i].remove('')
        rev_sents[i]=(rev_sents[i],[find_keyword_number(rev_sents[i],keyword),len(rev_sents[i])],0)
      for i,s in enumerate(unrev_sents):
        unrev_sents[i]=self.model.bpe_model.wp_sent(s).split(' ')
        while '' in unrev_sents[i]:
          unrev_sents[i].remove('')
        unrev_sents[i]=(unrev_sents[i],[[],len(unrev_sents[i])],1)
      if inc_bg:
        for i,s in enumerate(bg_sents):
          bg_sents[i]=self.model.bpe_model.wp_sent(s).split(' ')
          while '' in bg_sents[i]:
            bg_sents[i].remove('')
          bg_sents[i]=(unrev_sents[i],[[],len(bg_sents[i])],2)

    if self.use_model or use_bpe and not self.use_model:
      keyword=self.model.numerize_data(keyword.split(' '))
      while '' in keyword:
        keyword.remove('')

      for i,s in enumerate(rev_sents):
        s=s.split(' ')
        while '' in s:
          s.remove('')
        s=self.model.numerize_data(s)
        rev_sents[i]=(s,[find_keyword_number(s,keyword),len(s)],0)
      for i,s in enumerate(unrev_sents):
        s=s.split(' ')
        while '' in s:
          s.remove('')
        s=self.model.numerize_data(s)
        unrev_sents[i]=(s,[[],len(s)],1)
      if inc_bg:
        for i,s in enumerate(bg_sents):
          s=s.split(' ')
          while '' in s:
            s.remove('')
          s=self.model.numerize_data(s)
          bg_sents[i]=(s,[[],len(s)],2)

    else: #not use model and not use bpe
      keyword=clean(keyword).split(' ')
      while '' in keyword:
        keyword.remove('')

      self.no_model_wp2idx={}
      for i,s in enumerate(rev_sents):
        rev_sents[i]=clean(s).split(' ')
        while '' in rev_sents[i]:
          rev_sents[i].remove('')
        for w in rev_sents:
          if w in self.no_model_wp2idx:
            self.no_model_wp2idx[w]=len(self.no_model_wp2idx)+1

      for i,s in enumerate(unrev_sents):
        unrev_sents[i]=clean(s).split(' ')
        while '' in unrev_sents[i]:
          unrev_sents.remove('')
        for w in unrev_sents:
          if w in self.no_model_wp2idx:
            self.no_model_wp2idx[w]=len(self.no_model_wp2idx)+1

      if inc_bg:
        for i,s in enumerate(bg_sents):
          bg_sents[i]=clean(s).split(' ')
          while '' in bg_sents[i]:
            bg_sents[i].remove('')
          for w in bg_sents:
            if w in self.no_model_wp2idx:
              self.no_model_wp2idx[w]=len(self.no_model_wp2idx)+1

      self.no_model_idx2wp = dict(zip(self.no_model_wp2idx.values(), self.no_model_wp2idx.keys()))

      keyword=[self.no_model_wp2idx[w] for w in keyword]

      for i,s in enumerate(rev_sents):
        for j,w in enumerate(s):
          rev_sents[i][j]=self.no_model_wp2idx[w]
        rev_sents[i]=(rev_sents[i],[find_keyword_number(rev_sents[i],keyword),len(rev_sents[i])],0)
      
      for i,s in enumerate(unrev_sents):
        for j,w in enumerate(s):
          unrev_sents[i][j]=self.no_model_wp2idx[w]
        unrev_sents[i]=(unrev_sents[i],[[],len(unrev_sents[i])],1)
      
      if inc_bg:
        for i,s in enumerate(bg_sents):
          for j,w in enumerate(s):
            bg_sents[i][j]=self.no_model_wp2idx[w]
          bg_sents[i]=(bg_sents[i],[[],len(bg_sents[i])],2)
    
    self.rev_data=rev_sents
    self.unrev_data=unrev_sents
    self.bg_data=bg_sents

  def plot_sequence_length(self):
    all=[len(s[0]) for s in self.rev_data]
    if not self.data_is_text: all.extend([len(s[0]) for s in self.unrev_data])
    if self.bg_data is not None:
      all.extend([len(s[0]) for s in self.bg_data])
    
    count=collections.Counter(all)
    count=count.most_common(len(count))

    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.title('Length sequence')

    X,Y=[],[]
    for length,freq in count:
      X.append(length)
      Y.append(freq) 
    plt.scatter(X,Y)
    plt.show()

  def generate_data(self,train_ratio,inc_bg_ratio,max_length,min_length):
    assert max_length>0
    assert min_length<max_length and min_length>=0
    assert train_ratio >0 and train_ratio<1 and inc_bg_ratio >0 and inc_bg_ratio<1

    rev_data=[(x,y,z) for x,y,z in self.rev_data if y[-1]<=max_length and y[-1]>=min_length]
    unrev_data=[x for x in self.unrev_data if x is not None] if self.data_is_text else self.unrev_data
    unrev_data=[(x,y,z) for x,y,z in unrev_data if y[-1]<=max_length and y[-1]>=min_length]
    bg_data=[(x,y,z) for x,y,z in self.bg_data if y[-1]<=max_length and y[-1]>=min_length] if self.bg_data is not None else None

    if self.bg_data is None: inc_bg_ratio=0
    assert len(rev_data)>0
    
    if inc_bg_ratio>0:  
      bg_total=len(bg_data)
      bg_size=int(len(bg_data)*inc_bg_ratio)
      bg_data=bg_data[:bg_size]
      print('background data: {} adopted: {}'.format(bg_total,bg_size))
      unrev_data.extend(bg_data)
    
    random.shuffle(unrev_data)
    random.shuffle(rev_data)
    print('total rev data: {} total unrev data: {}'.format(len(rev_data),len(unrev_data)))
    
    rev_ratio=int(train_ratio*len(rev_data))
    unrev_ratio=int(train_ratio*len(unrev_data))

    train_data=rev_data[:rev_ratio]
    train_data.extend(unrev_data[:unrev_ratio])

    test_data=rev_data[rev_ratio:]
    test_data.extend(unrev_data[unrev_ratio:])

    random.shuffle(train_data)
    random.shuffle(test_data)
    
    train_input,train_dec_input,train_group=map(list, zip(*train_data))
    test_input,test_dec_input,test_group=map(list, zip(*test_data))

    train_target=[0]*len(train_input)
    test_target=[0]*len(test_input)

    #we should also consider doing valid_weight at the building data step, but there might be memory consideration
    for i,x in enumerate(train_dec_input):
      train_target[i]=[0.9,0.1] if train_group[i]==0 else [0.1,0.9] #there might be a smarter way?
      x=valid_weight(keywords=x[0],seqlen=x[-1],maxlen=max_length) if len(x)==2 else valid_weight(bases=x[0],chunk_size=x[1],seqlen=x[-1],maxlen=max_length)
      train_dec_input[i]=np.array(x)[np.newaxis].T
      
    for i,x in enumerate(test_dec_input):
      test_target[i]=[0.9,0.1] if test_group[i]==0 else [0.1,0.9]
      x=valid_weight(keywords=x[0],seqlen=x[-1],maxlen=max_length) if len(x)==2 else valid_weight(bases=x[0],chunk_size=x[1],seqlen=x[-1],maxlen=max_length)
      test_dec_input[i]=np.array(x)[np.newaxis].T

    for i,s in enumerate(train_input):
      s.extend([0 for i in range(max_length-len(s))])
      train_input[i]=np.array(s)[np.newaxis].T
    for i,s in enumerate(test_input):
      s.extend([0 for i in range(max_length-len(s))])
      test_input[i]=np.array(s)[np.newaxis].T

    print('{}:{}\n{}:{}'.format(np.sum([np.argmax(x) for x in test_target]),len(test_target),np.sum([np.argmax(x) for x in train_target]),len(train_target))) 
    assert np.sum([np.argmax(x) for x in test_target])<len(test_target) and np.sum([np.argmax(x) for x in train_target])<len(train_target)
    return train_input, test_input, train_dec_input, test_dec_input, train_target, test_target, train_group, test_group