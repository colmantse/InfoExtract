'''
This program read in a text and performs concordance upon it
Note this is done without tokenization
Author: Colman Tse
Data: 31/jul/17
'''

import logging,re,argparse, sys

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Concordancer')
parser.add_argument('text',action="store",help='the namepath of the text upon concordance should perform')
parser.add_argument('keyword', action="store",help='the target string')
parser.add_argument('window_size', action="store",type=int,help='window_size')

#define text obj 
class text_holder:
  def __init__(self,filename):
    with open(filename,'r',encoding='utf8') as f:
      self.text=f.read()
    logger.info('Data_size: {}'.format(len(self.text)))

  def write_concord(self,keyword,window_size,result_to):
    ls=[(m.start(0), m.end(0)) for m in re.finditer(keyword,self.text)]
    logger.info('Found {} occurences'.format(len(ls)))
    with open(result_to,'w',encoding='utf8') as f:
      logger.info('Writing...')
      for i,(head,tail) in enumerate(ls):
        head=max(head-window_size,0)
        tail=min(tail+window_size,len(self.text))
        f.write(str(i)+'  '+clean(self.text[head:tail])+'\n\n')

#clean function for neat presentation
def clean(string):
  return re.sub(r'\s+',' ',string)

def main():
  args=parser.parse_args()
  k=args.text.rfind("\\")
  l=args.text.rfind(".")
  result_name='result{}_{}_{}.txt'.format(args.text[k:l],args.keyword,args.window_size)
  holder=text_holder(args.text)
  holder.write_concord(args.keyword,args.window_size,result_name)
  
if __name__ == "__main__":
  main()