#please switch to chcp 65001 
#init
import os,re,webbrowser, logging
from itertools import chain
from collections import Counter
from gensim.summarization import textcleaner

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

#Define court object

class court:
  def __init__(self, cases=[]):
    assert isinstance(cases,list)
    if len(cases)>0:
      assert isinstance(cases[0],str) 
    self.cases=cases
    self.noc=len(cases)
    self.keyword=None
    self.window_size=None
    self.sents=None

  def check_number(self, number):
    guard=number in range(0,self.noc)
    if guard is False:
      logger.error('case does not exist, please check the number of cases the court possesses.')
    return guard

  def check_case_proper(self, case):
    guard=type(case) is str and case is not None
    if guard is False:
      logger.error('case cannot be non-string and cannot be null')
    return guard

  def check_keyword(self):
    if self.keyword is None:
      self.keyword = input('assign keyword:')

  def check_window_size(self):
    if self.window_size is None:
      self.window_size = int(input('assign window size:'))

  def number_of_cases(self):
    print(self.noc)

  def copyTo(self,court,number):
    court.add(self.cases[number])

  def cutTo(self,court,number):
    if court.add(self.cases[number]):
      self.remove(number)

  def cutAll(self, court):
    for i in range(self.noc):
      self.cutTo(court,0)

  def copyAll(self, court):
    for i in range(self.noc):
      self.copyTo(court,i)

  def cancel(self, court):
    for x in court.cases:
      if x in self.cases:
        self.remove(self.cases.index(x))

  def add(self,case):
    guard=self.check_case_proper(case)
    if guard:
      self.cases.append(case)
      self.noc+=1
    return guard

  def remove(self,number):
    if self.check_number(number):
      del self.cases[number]
      self.noc-=1

  def display_case(self,number):
    if self.check_number(number):
      print(self.cases[number])

  def get_url(self,number):
    case=self.cases[number]
    url=case[case.index('URL:'):case.index('.html')] if ('.html') in case[case.index('URL:'):] else case[case.index('URL:'):]
    return url+'.html'

  def write(self,name):
    with open(name,'w',encoding='utf8') as f:
      for case in self.cases:
        f.write(case+'.html\n/n/n')

  def write_urls(self,name):
    with open(name,'w',encoding='utf8') as f:
      for i in range(self.noc):
        f.write(self.get_url(i)+'\n')

  def write_quotes(self,name):
    if self.sents is None:
      self._clean_sentences()
    self.check_keyword()
    with open(name, 'w',encoding='utf8') as f:
      for i,sents in enumerate(self.sents):
        for sent in sents:
          if self.keyword in sent:
            f.write('Case: {} {}\n'.format(i,sent))

  def goto_link(self, number):
    if self.check_number(number):
      url=self.get_url(number)[5:]
      webbrowser.open(url,2)

  def display_quote(self,number,display_sentence=False):
    if display_sentence:
      self.check_keyword()
      if self.sents is None:
        self._clean_sentences()  
      for sent in self.sents[number]:
        if self.keyword in sent:
          print(sent+'\n')
    else:
      self._concord(number)

  def search(self, keyword):
    result=court([x for x in self.cases if keyword in x])
    result.keyword=keyword
    return result

  def _clean_sentences(self):
    self.sents=[]
    print('cleaning {} cases'.format(self.noc))
    for case in self.cases:
      case=case.replace('/n','')
      sentences=textcleaner.split_sentences(case)
      count=5
      if not ('___' in sentences[count] or '---' in sentences[count]):
        count-=1
      sentences=sentences[count+1:-6]
      self.sents.append(sentences)

  def _concord(self,number):
    self.check_keyword()
    self.check_window_size()
    
    ls=[(m.start(0), m.end(0)) for m in re.finditer(self.keyword,self.cases[number])]
    for (head,tail) in ls:
      head=head-self.window_size*5
      tail=tail+self.window_size*5
      head=head if head>0 else 0
      tail=tail if tail<len(self.cases[number]) else len(self.cases[number])-1
      print(self.cases[number][head:tail]+'\n')

#simple_op

def load(name):
  with open(name,'r',encoding='utf8') as f:
    tmp=f.read()
  logger.info(name+' loaded.')
  obj = court(tmp.split('.html\n/n/n'))
  for (i,x) in enumerate(obj.cases):
    if len(x)==0:
      obj.remove(i)
  return obj

def load_all(list_to_load):
  ls= [os.stat(x).st_size for x in list_to_load]
  hold=load(list_to_load[ls.index(max(ls))])
  del list_to_load[ls.index(max(ls))]
  
  for x in list_to_load:
    tmp=load(x)
    tmp.cutAll(hold)
  return hold

def organize():
  import organize
