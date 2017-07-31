import re,zipfile,random
from nltk import word_tokenize
'''
Cleaning the sequences leaving only alphanumeric characters
'''
def clean(str):
  pattern=re.compile('[\\W_]+')
  return pattern.sub(' ',str)

'''
sometimes cleaning will cause memory error
'''
def clean_text(str):
  pattern=re.compile('[\\W_]+')
  text=''
  while len(str)!=0:
    text+=pattern.sub(' ',str[:10000])
    str=str[10000:]
  return text

'''
this function generate unnatural sequences with single invalidity
maynot be the fastest, might consider enumerate, though am not sure of the logic
'''
def randomize_sequence(seq):
  seq_len=len(seq)
  pos=[random.randint(0,seq_len),random.randint(0,seq_len)]
  pos.sort()
    
  base=seq_len-5 if pos[0]>=seq_len-5 and seq_len>5 else pos[0] 
    
  chunk_size=random.randint(pos[0],pos[1])
  if base+chunk_size>=seq_len and chunk_size>3:
    chunk_size= 3
   
  new_seq=[0]*seq_len
  for i in range(base): 
    new_seq[i]=seq[i]
  for i in range(base+chunk_size,seq_len):
    new_seq[i-chunk_size]=seq[i]
  i=0
  while i<chunk_size and seq_len-chunk_size+i>=0 and base+i<seq_len:
    new_seq[seq_len-chunk_size+i]=seq[base+i]
    i+=1

  if str(seq)!=str(new_seq):
    return (new_seq,[base,chunk_size,len(seq)])
  return None

'''
this function is for decoder input: it produces a vector with 0.8 as most valid 
and 0.2 as most invalid, we assume there is only 1 transposition
'''

def valid_weight(bases=None,chunk_size=None,keywords=None,seqlen=None,maxlen=None):
  assert (bases is not None and keywords is None) or (bases is None and keywords is not None)
  assert seqlen is not None and maxlen is not None  

  #locate keyword attention
  def locate_key(pos):
    for i in range(1,seqlen):
      w=0.1+0.1*(i//3) if 0.1+0.1*(i//3)<0.6 else 0.55
      w_seq[pos]=0.1
      if pos-i >= 0 and w>w_seq[pos-i]:
        w_seq[pos-i]=w
      if pos+i < seqlen and w>w_seq[pos+i]:
        w_seq[pos+i]=w

  #identify transpose position
  def transpose_weight(trans_pos):
    for i in range(1,4):
      w=0.2*i
      pos=trans_pos-i
      if pos >=0 and w<w_seq[pos]:
        w_seq[pos]=w
      pos=trans_pos+i-1
      if pos <seqlen and w<w_seq[pos]:
        w_seq[pos]=0.2*i 

  w_seq=[0.8]*seqlen

  #for now we only assume 1 invalidity transposition per sequence
  if bases is not None and chunk_size>0:
    transpose_a=bases
    transpose_b=bases+seqlen-(bases+chunk_size)
    transpose_weight(transpose_a)
    transpose_weight(transpose_b)

  if keywords is not None:
    for key in keywords:
      locate_key(key)

  w_seq.extend([0 for _ in range(maxlen-seqlen)])
  return w_seq

'''
loading operations
'''

def load_data(filename, encode='utf8', mode='sequence'):
  '''
  This function reads a file line by line and output list of sequence/tokenized_sequence/tokens
  '''
  ls=[]
  with open(filename, 'r', encoding=encode) as f:
    for line in f:
      if mode=='sequence':
        ls.append(line[:-1])
      if mode=='tokenized_sequence':
        ls.append(word_tokenize(line))
      if mode=='token':
        ls.extend(word_tokenize(line))
  return ls

def read_zip_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  print('Data size:',len(data))
  return data

def read_data(filename):
  with open(filename, 'r', encoding='utf8') as f:
    data=f.read().split()
  print('Data size:',len(data))
  return data

def read_large_data(filename):
  with open(filename, 'r', encoding='utf8') as f:
    raw=f.read()
    read=[]
    while len(raw)!=0:
      read.extend(raw[:100000000].split())
      raw=raw[100000000:]
    print('Data size:',len(read))
    return read

'''
Just to download toy data in case
#filename = maybe_download('data/text8.zip', 31344016)
'''
def maybe_download(filename, expected_bytes):
  url = 'http://mattmahoney.net/dc/'
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

