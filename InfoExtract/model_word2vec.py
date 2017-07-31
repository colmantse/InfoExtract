import collections, math, os, random
import numpy as np, tensorflow as tf
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

class word2vec_model:

  def __init__(self,bpe_model=None,use_bpe=False):
    self.bpe_model=bpe_model
    self.use_bpe=use_bpe

  '''
  below skipgram implementation is adapted from 
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb
  udacity's tensorflow tutorial on deeplearning word2vec skipgram model
  '''

  def build_dataset(self,words,vocab_size=50000,min_count=None):
    #take in list of token as processed like text8
    self.data=words
    if self.use_bpe:
      self.data=[]
      while len(words)!=0: #use while loop to swap words with data, memory efficient
        self.data.extend([wp for wp in self.bpe_model.wp_word('_'+words[0])])
        del words[0]

    v_size = self.bpe_model.active_nodes if self.use_bpe else vocab_size
    count = collections.Counter(self.data)
    count = count.most_common(v_size) 
    if min_count is not None:
      count={wp:f for wp,f in count if f>min_count}
    self.wp2idx = {'UNK':1}
    for wp,_ in count:
      self.wp2idx[wp]=len(self.wp2idx)+1 #we want to reserve 0 as padding
    self.idx2wp = dict(zip(self.wp2idx.values(), self.wp2idx.keys()))
    
    data=[]
    while len(self.data)!=0:
      if self.data[0] not in self.wp2idx:
        data.append(self.wp2idx['UNK'])
      else:
        data.append(self.wp2idx[self.data[0]])
      del self.data[0]
    
    self.data=data  

    print('data size', len(self.data))
    print('Sample data', self.data[:10])
    print('data:', [self.idx2wp[di] for di in self.data[:8]])

  def numerize_data(self,ls_of_string):
    tmp=['0']*len(ls_of_string)
    for i,wp in enumerate(ls_of_string):
      if wp not in self.wp2idx:
        tmp[i]=self.wp2idx['UNK']
      else:
        tmp[i]=self.wp2idx[wp]
    return tmp

  def init_train(self,batch_size=200,embedding_size=300,skip_window=1,num_skips=2,num_sampled=400, use_skipgram=True):
    '''
    remember that batch_size is context_size/window_size for cbow, do not forget to adjust
    '''    
    if not use_skipgram:
      assert batch_size%2==1 #the 1 left is the middle word
    self.data_index=0 #for data_indexing in generate_batch
    self.batch_size=batch_size
    self.embedding_size=embedding_size
    self.use_valid=False #set to True using use_valid to visualize training progress
    self.use_skipgram=use_skipgram #by default use skipgram, if switched False, use CBOW
    self.skip_window=skip_window if use_skipgram else None # How many words to consider left and right.
    self.num_skips=num_skips if use_skipgram else None  # How many times to reuse an input to generate a label.
    self.num_sampled=num_sampled # Number of negative examples to sample.

  def view_valid(self,valid_size=16,valid_window=100):
    ''' 
    We pick a random validation set to sample nearest neighbors. here we limit the
    validation samples to the words that have a low numeric ID, which by
    construction are also the most frequent.
    ''' 
    self.valid_size=valid_size # Random set of words to evaluate similarity on.
    self.valid_window=valid_window # Only pick dev samples in the head of the distribution.
    self.valid_examples=np.array(random.sample(range(valid_window), valid_size))
    self.use_valid=True

  def generate_batch(self):
    '''
    from udacity's tensorflow skipgram tutorial
    '''
    
    def skip_gram_batch():
      batch_size, num_skips, skip_window=self.batch_size, self.num_skips, self.skip_window
      
      assert batch_size % num_skips == 0
      assert num_skips <= 2 * skip_window
      batch = np.ndarray(shape=(batch_size), dtype=np.int32)
      labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
      span = 2 * skip_window + 1 # [ skip_window target skip_window ]
      buffer = collections.deque(maxlen=span)
      for _ in range(span):
        buffer.append(self.data[self.data_index])
        self.data_index = (self.data_index + 1) % len(self.data)
      for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]	
        for j in range(num_skips):
          while target in targets_to_avoid:
            target = random.randint(0, span - 1)
          targets_to_avoid.append(target)
          batch[i * num_skips + j] = buffer[skip_window]
          labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(self.data[self.data_index])
        self.data_index = (self.data_index + 1) % len(self.data)
      return batch, labels

    def cbow_batch():
      batch_size=self.batch_size
      assert batch_size%2==1 #the 1 left is the middle word
      
      batch=[]
      for i in range(batch_size):
        if i==batch_size//2:
          continue
        batch.append(self.data[self.data_index])
        self.data_index = (self.data_index + 1) % len(self.data)
      batch=np.array(batch)
      label=np.array(self.data[(self.data_index+batch_size)%len(self.data)])[np.newaxis][np.newaxis]
      return batch, label

    batch,labels=skip_gram_batch() if self.use_skipgram else cbow_batch()
    return batch,labels

  def train(self,num_steps,save=None):
    graph = tf.Graph()
    with graph.as_default():#, tf.device('/cpu:0'):
      train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size]) if self.use_skipgram else tf.placeholder(tf.int32, shape=[self.batch_size-1])
      train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1]) if self.use_skipgram else tf.placeholder(tf.int32, shape=[1,1])
      valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32) if self.use_valid else None

      embeddings = tf.Variable(tf.random_uniform([len(self.wp2idx), self.embedding_size], -1.0, 1.0))
      softmax_weights = tf.Variable(tf.truncated_normal([len(self.wp2idx), self.embedding_size],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
      softmax_biases = tf.Variable(tf.zeros([len(self.wp2idx)]))

      embed = tf.nn.embedding_lookup(embeddings, train_dataset)
      embed = embed if self.use_skipgram else tf.reshape(tf.reduce_mean(embed,0),[1,-1])
      loss = tf.reduce_mean(
               tf.nn.sampled_softmax_loss(
                 weights=softmax_weights, biases=softmax_biases, inputs=embed,
                 labels=train_labels, num_sampled=self.num_sampled, num_classes=len(self.wp2idx)))

      optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset) if self.use_valid else None
      similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings)) if self.use_valid else None

    saver = tf.train.Saver()

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('initialized')
      average_loss=0
      for step in range(num_steps):
        batch_data, batch_labels = self.generate_batch()
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        if len(batch_data)!=self.batch_size-1:
          print(batch_data)
        _, l = session.run([optimizer,loss],feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
          if step > 0:
            average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
            # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          if not self.use_valid:
            continue
          sim = similarity.eval()
          for i in range(self.valid_size):
            valid_word = self.idx2wp[self.valid_examples[i]]
            top_k = 8 # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k+1]
            log = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              close_word = self.idx2wp[nearest[k]]
              log = '%s %s,' % (log, close_word)
            print(log)
      self.final_embeddings = normalized_embeddings.eval()
      
      if saving: #probably unnecessary, the model holds a copy of the matrix anyway
        print("Saving as",save)
        saver.save(session, save)

  def plot_embedding(self,num_points=400):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(self.final_embeddings[1:num_points+1, :])

    def plot(embeddings, labels):
      assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
      pylab.figure(figsize=(15,15))  # in inches
      for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
      pylab.show()
     
    words = [self.idx2wp[i] for i in range(1, num_points+1)]
    plot(two_d_embeddings, words)

