import re, collections, time
from nltk import word_tokenize

class wp_node:
  def __init__(self,wp='',is_active=False, parent=None):
    self.node=wp
    self.is_active=is_active
    self.active_nodes=0
    self.parent=parent
    self.children=[]

  """equal methods implementation adapted from:
  http://stackoverflow.com/questions/390250/elegant-ways-to-support-equivalence-equality-in-python-classes
  """
  def __eq__(self, other):
    """Override the default Equals behavior"""
    if isinstance(other, self.__class__):
      return self.node == other.node
    return NotImplemented

  def __ne__(self, other):
    """Define a non-equality test"""
    if isinstance(other, self.__class__):
      return not self.__eq__(other)
    return NotImplemented

  def __hash__(self):
    """Override the default hash behavior (that returns the id or the object)"""
    return hash(tuple(sorted(self.__dict__.items())))

  def has_parent(self):
    return self.parent is not None

  def has_child(self):
    return len(self.children)>0

  def update_Activity(self):
    if self.has_parent():
      self.parent.active_nodes+=1
      self.parent.update_Activity()

  def turn_active(self):
    if not self.is_active:
      self.is_active=True
      self.update_Activity()

  def check_unit(self,wp):
    if isinstance(wp,str):
      wp=wp_node(wp)
    if len(wp.node)>1:
      head,tail=wp_node(wp.node[0]),wp_node(wp.node[1:])
      if head not in self.children:
        return False
      child=self.children[self.children.index(head)]
      return True and child.check_unit(tail)
    if len(wp.node)==1:
      if wp in self.children and self.children[self.children.index(wp)].is_active:
        return True
    return False

  def add(self,wp):
   
    if len(wp.node)>1:
      head,tail = wp_node(wp.node[0],parent=self),wp_node(wp.node[1:])
      if head not in self.children:
        self.children.append(head)
      self.children[self.children.index(head)].add(tail)  

    if len(wp.node)==1:
      if wp in self.children:
        self.children[self.children.index(wp)].turn_active()
      else:
        wp.parent=self
        wp.turn_active()
        self.children.append(wp)

  def extends(self, nodes):
    for node in nodes:
      self.add(node)

  def init_symbols(self, vocab):
    init_sym=[]
    for word in vocab:
      sym=word.split()
      init_sym.extend(sym)
      init_sym=list(set(init_sym))
    self.extends([wp_node(sym) for sym in init_sym])

  def extract_units(self,prev_str=''):
    tmp_str=prev_str+self.node
    
    if not self.has_child and self.is_active:
      return [tmp_str]
    
    if self.has_child:
      ls=[tmp_str] if self.is_active else []
      for child in self.children:
        ls.extend(child.extract_units(tmp_str))
      return ls  
  
  def wp_word(self, word):
    """Take in a string of word and break it into word pieces"""
    n=len(word)
    for i in range(n):
      for j in range(n):
        if n-j>=n-i:
          pre=word[:j]
          unit=word[j:j+n-i]
          post=word[j+n-i:]
          #print('{} {} {}'.format(pre,unit,post))
          if len(unit)>0 and self.check_unit(wp_node(unit)):
            ls=self.wp_word(pre)
            ls.extend([unit])
            ls.extend(self.wp_word(post))
            return ls
        else:
          break
    return []

  def wp_sent(self,sent):
    if isinstance(sent,list):
      sent=' '.join(sent)
    sent=word_tokenize(sent)
    sent=['_'+x for x in sent]
    return ' '.join([' '.join(self.wp_word(x)) for x in sent])


'''operations'''

def spacify_vocab(vocab):
  ref_ls=list(vocab)
  ref_ls=[x.replace(' ','') for x in ref_ls if '_' not in x]
  modify_ls=[x for x in ref_ls]
  modify_ls=['_'+' '.join(list(x)) for x in modify_ls]
  return {modify_ls[i]:vocab[ref_ls[i]] for i in range(len(ref_ls)) if '_.' not in modify_ls[i]}

"""get_stats and merge_vocab adapted from:
https://github.com/rsennrich/subword-nmt/blob/master/

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

def get_stats(vocab):
  pairs = collections.defaultdict(int)
  for word, freq in vocab.items():
    symbols = word.split()
    for i in range(len(symbols)-1):
      pairs[symbols[i],symbols[i+1]] += freq
  return pairs

def merge_vocab(pair, v_in, lookup):
  v_out = {}
  bigram = ' '.join(pair)
  p=re.compile(bigram)
  for word in v_in:
    w_out = p.sub(''.join(pair), word)
    v_out[w_out] = v_in[word]
    
    hasSpace=' ' in word
    noSpace=' ' not in w_out
    if hasSpace and noSpace:
      lookup.add(wp_node(w_out))
      
  return v_out, lookup

def merge_vocabs(pairs, v_in, lookup):
  v_out = {}
  bigram_ls = [' '.join(pair) for pair in pairs]
  p_ls=[re.compile(bigram) for bigram in bigram_ls]
  for word in v_in:
    w_out=word
    for p,pair in zip(p_ls,pairs):
      w_out = p.sub(''.join(pair), w_out)  
    v_out[w_out] = v_in[word]

    hasSpace=' ' in word
    noSpace=' ' not in w_out
    if hasSpace and noSpace:
      lookup.add(wp_node(w_out))

  return v_out, lookup

def gather_pairs(pairs):
  gathered=''
  g_ls=[]
  while True:
    bigram = find_best(pairs)
    if bigram is None:
      break
    first_gram, second_gram = bigram
    if first_gram in gathered or second_gram in gathered:
      break
    gathered+=' '.join(bigram)
    g_ls.append(bigram)
  return g_ls

def find_best(pairs):
  best = None
  while best is None or '.' in ''.join(best):
    if len(pairs)==0:
      best=None
      break
    best = max(pairs, key=pairs.get)
    pairs.pop(best)
  return best

def wordpiece(vocab, iteration=1000, units=None, display=False, until_converge=False, min_threshold=500):
  assert iteration>=0 and units is None or units>=0
  vocab={x:vocab[x] for x in vocab if vocab[x]>min_threshold}
  vocab=spacify_vocab(vocab)
  lookup=wp_node()
  lookup.init_symbols(vocab)  
  time1=time.time()  
  i=0
  while until_converge or units is None and i < iteration or units is not None and units>lookup.active_nodes:
    pairs = get_stats(vocab)
    if display is False:  
      bestpairs=gather_pairs(pairs)
      if len(bestpairs)==0:
        print('bpe converged after {} iterations'.format(i))
        break
      vocab,lookup = merge_vocabs(bestpairs, vocab, lookup)
      lookup.extends([wp_node(''.join(best)) for best in bestpairs])
    else:
      best=find_best(pairs)
      if best is None:
        print('bpe converged after {} iterations'.format(i))
        break
      vocab,lookup = merge_vocab(best,vocab,lookup)
      lookup.add(wp_node(best))
      print(str(i)+' '+' '.join(best))  
    i+=1
    #print('{}|{}'.format(i,lookup.active_nodes))
  time2=time.time()
  print('wordpiece model built in {}s, {} wordpieces collected.'.format(int(time2-time1), lookup.active_nodes))
  return lookup
