import tensorflow as tf
import pickle,time
from modified_dynamic import bidirectional_dynamic_rnn, dynamic_rnn
from tensorflow.python.util import nest

'''
TODO: 
2. Reimplement all states option
3. Implement mask loss and seq2seq output
'''

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def _get_target_from_dec_input(tg):
  #shape = batch_size, max_length, dimension
  tg = tf.scan(lambda a, _:tf.round(a), tg)
  return tg

def _get_last_relevant(outputs,seq_len):
  batch_size=tf.shape(outputs)[0]
  max_length=outputs.get_shape()[1]
  num_hidden=tf.shape(outputs)[2]
  index = tf.range(0, batch_size) * max_length + (seq_len - 1)
  flat = tf.reshape(outputs, [-1, num_hidden])
  op = tf.gather(flat, index) #last relevant output
  return op
  
class classifier:
  
  def __init__(self,model=None, use_model=False):
    self.model=model
    self.use_model=use_model
    
    if use_model is True:
      assert model is not None

  def create_structure(self,num_classes=2,drop_out=0.6,num_hidden=1000,number_of_layers=4,
                       use_bidirection=False,
                       use_decoder=False,
                       use_attention_decoder=False, 
                       classify_seq=True, 
                       seq2seq=False, 
                       seq_filter_weight=0.5,
                       learning_rate=0.3,max_length=100,min_length=0):

    self.graph = tf.Graph()
    with self.graph.as_default():
      self.seq2seq=False
      self.classify_seq=False

      assert not (classify_seq is True and classify_seq is seq2seq) #the structure is either a sequence classifier or a seq2seq model or neither
      if seq2seq: self.seq2seq=True
      if classify_seq: self.classify_seq=True

      if use_attention_decoder:
        use_bidirection=True
        use_decoder=True

      assert seq_filter_weight>=0 and seq_filter_weight<=1

      self.max_length=max_length
      self.min_length=min_length
      self.data = tf.placeholder(tf.int32, [None,max_length,1],name='input_data') if self.use_model else tf.placeholder(tf.float32, [None,max_length,1],name='input_data')#Batch_size = None means unknown
      self.decoder_input = tf.placeholder(tf.float32, [None,max_length,1],name='weighted_data')

      if not seq2seq:
        self.target = tf.placeholder(tf.float32, [None, num_classes],name='output_class')
      else:
        self.target = tf.placeholder(tf.float32, [None,max_length,1],name='output')
    
      self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

      seq_len=length(self.data)

      #Embedding matchup
      if self.use_model:
        with tf.name_scope('match_embedding'):
          W = tf.Variable(tf.constant(0.0, shape=[len(self.model.final_embeddings),len(self.model.final_embeddings[0])]),trainable=False,name="W")
          self.embedding_placeholder = tf.placeholder(tf.float32, [len(self.model.final_embeddings), len(self.model.final_embeddings[0])])
          self.embedding_init = W.assign(self.embedding_placeholder)
          lookup_data = tf.nn.embedding_lookup(W, self.data)
          data = tf.reshape(lookup_data,[-1,int(lookup_data.get_shape()[1]),int(lookup_data.get_shape()[3])])

      with tf.variable_scope('encoder'):
        enc_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True) for _ in range(number_of_layers)] , state_is_tuple=True) if number_of_layers>1 else tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
      
        if drop_out>0: #use dropout
          enc_cell=tf.contrib.rnn.DropoutWrapper(enc_cell,output_keep_prob=drop_out)
        
        enc_outputs=None
        enc_input=data if self.use_model else self.data
      
        if use_bidirection:
          with tf.variable_scope('forward'):
            fw_cell = tf.contrib.rnn.LSTMCell(num_hidden)
          with tf.variable_scope('backward'):
            bw_cell = tf.contrib.rnn.LSTMCell(num_hidden)
          _,_, bi_all_states = bidirectional_dynamic_rnn(fw_cell,bw_cell,enc_input,dtype=tf.float32,sequence_length=seq_len)
          bi_all_states=tf.concat(bi_all_states,2)
          enc_outputs,enc_state,all_states = dynamic_rnn(enc_cell,bi_all_states,dtype=tf.float32,sequence_length=seq_len)
        else:
          enc_outputs,enc_state = tf.nn.dynamic_rnn(enc_cell,enc_input,dtype=tf.float32,sequence_length=seq_len)         

      if use_decoder:
        with tf.variable_scope('decoder'):
          dec_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True) for _ in range(number_of_layers)], state_is_tuple=True) if number_of_layers>1 else tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
            
          if drop_out>0: #use dropout
            dec_cell=tf.contrib.rnn.DropoutWrapper(dec_cell,output_keep_prob=drop_out)

          dec_input=self.decoder_input
          encoder_all_states = all_states if use_attention_decoder else None
          if use_attention_decoder: enc_state=None
          dec_outputs,dec_state,_ = dynamic_rnn(dec_cell,dec_input,dtype=tf.float32,
                                    sequence_length=seq_len, initial_state=enc_state, 
                                    encoder_all_states=encoder_all_states)

      outputs = dec_outputs if use_decoder else enc_outputs
    
      if classify_seq: #if this is a sequence classifier
        # batch x classes
        target=self.target
        op=_get_last_relevant(outputs,seq_len)
  
      elif seq2seq: #if this is a seq2seq model
        target=self.target
        op=outputs
  
      else: #if this is a sequence filter, specialize in picking out subsequence of interest
        target=_get_target_from_dec_input(self.decoder_input)
        d_target=self.target
        # batch x time x classes
        ops=outputs
        with tf.variable_scope('downstream_classifier'):
          downstream_cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
          if drop_out>0: #use dropout
            downstream_cell=tf.contrib.rnn.DropoutWrapper(downstream_cell,output_keep_prob=drop_out)
          downstream_outputs,_=tf.nn.dynamic_rnn(downstream_cell,ops,dtype=tf.float32,sequence_length=seq_len)
        d_op=_get_last_relevant(downstream_outputs,seq_len) #we cannot use the non-zero pad trick since seq_filter uses only 0 and 1	

      weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[-1])]),name='weight')
      bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[-1]]),name='bias')

      if not classify_seq: #if this is not a sequence classifier
        op = tf.reshape(ops,[-1,num_hidden])
        target = tf.reshape(target,[-1,int(target.get_shape()[-1])])

      logits=tf.nn.xw_plus_b(op,weight,bias)
      cost=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target)
    
      if not classify_seq and not seq2seq:
        d_w=tf.Variable(tf.truncated_normal([num_hidden, int(d_target.get_shape()[-1])]),name='d_weight')
        d_b=tf.Variable(tf.constant(0.1, shape=[d_target.get_shape()[-1]]),name='d_bias')
        d_logits=tf.nn.xw_plus_b(d_op,d_w,d_b)
        d_cost=tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=d_target)

        w1,w2=seq_filter_weight,1-seq_filter_weight #the emphasis we wish to put on the filter task and the classification task
        self.prediction=tf.nn.softmax(logits),tf.nn.softmax(d_logits)
        self.loss=w1*tf.reduce_mean(d_cost)+w2*tf.reduce_sum(cost)
        mistakes = tf.not_equal(tf.argmax(d_target, 1), tf.argmax(tf.nn.softmax(d_logits), 1))
      else: #yet to implement bleu score evaluation for seq2seq
        self.prediction = tf.nn.softmax(logits)	
        self.loss = tf.reduce_sum(cost,name='loss') if not classify_seq else tf.reduce_mean(cost,name='loss')
        mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(tf.nn.softmax(logits), 1))    

      if not seq2seq:
        self.error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
	
      #learning_rate = tf.train.exponential_decay(0.1, global_step, 500, 0.7, staircase=True)
      with tf.name_scope("train") as scope:
        self.minimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss,global_step=self.global_step)

      self.init_op = tf.global_variables_initializer()
	  
      with tf.name_scope("summaries"):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("error", self.error)
        tf.summary.histogram("histogram_loss", self.loss)
        self.summary_op = tf.summary.merge_all()

  def train(self,batch_size,epoch,train_ratio=0.66,inc_bg_ratio=0,record_interval=None,resume=False,graph_directory='',data_path=None):
    
    def acquire_eval_list_location(train_group,test_group): 
      '''This function create 6 data list, rev, unrev, background for train and test dataset'''
      train_rev_location,train_unrev_location,train_bg_location=[],[],[]
      for i,x in enumerate(train_group):
        if x==0:
          train_rev_location.append(i)
        if x==1:
          train_unrev_location.append(i)
        if x==2:
          train_bg_location.append(i)
      test_rev_location,test_unrev_location,test_bg_location=[],[],[]
      for i,x in enumerate(test_group):
        if x==0:
          test_rev_location.append(i)
        if x==1:
          test_unrev_location.append(i)
        if x==2:
          test_bg_location.append(i)
      self.auxiliary_locations=[train_rev_location,train_unrev_location,train_bg_location,test_rev_location,test_unrev_location,test_bg_location]

    def acquire_eval_lists(data_s,location_list):
      '''receive a list of data as input'''
      
      def acquire_eval_list(data,location_list):
        ls=[0]*len(location_list)
        for i,x in enumerate(location_list):
          ls[i]=data[x]
        return ls

      return [acquire_eval_list(data,location_list) for data in data_s]

    def write_interval_summary(filewriter,tmp_data,step):
      summary = self.sess.run(self.summary_op,{self.data:tmp_data[0],self.decoder_input:tmp_data[1],self.target:tmp_data[2]})
      filewriter.add_summary(summary,global_step=step)	  

    def define_writers():
      #defining writer	  
      self.main_writer = tf.summary.FileWriter(graph_directory+"main/", self.graph)

      if not self.seq2seq:
        #auxiliary writers
        train_rev_writer = tf.summary.FileWriter(graph_directory+"train/rev/", self.graph)
        train_unrev_writer = tf.summary.FileWriter(graph_directory+"train/unrev/",self.graph)
        train_background_writer = tf.summary.FileWriter(graph_directory+"train/background/",self.graph)
        test_rev_writer = tf.summary.FileWriter(graph_directory+"test/rev/",self.graph)
        test_unrev_writer = tf.summary.FileWriter(graph_directory+"test/unrev/",self.graph)
        test_background_writer = tf.summary.FileWriter(graph_directory+"test/background/",self.graph)
        self.auxiliary_writers=[train_rev_writer,train_unrev_writer,train_background_writer,test_rev_writer,test_unrev_writer,test_background_writer]

    train_input,test_input,train_dec_input, test_dec_input, train_output,test_output,train_group,test_group = pickle.load(open(data_path,'rb')) if data_path is not None else self.generate_data(train_ratio,inc_bg_ratio) 
    no_of_batches = int(len(train_input)/batch_size)
    if not self.seq2seq: acquire_eval_list_location(train_group,test_group)

    define_writers()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),graph=self.graph)
    self.sess.run(self.init_op)


    #assigning embedding
    if self.use_model:
      self.sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: self.model.final_embeddings})

    #start training
    for i in range(epoch):
      ptr=0
      time_init=time.time()
      for j in range(no_of_batches):
        inp,dec,out=train_input[ptr:ptr+batch_size],train_dec_input[ptr:ptr+batch_size],train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        if (i*no_of_batches+j)%500 ==0:
          _,summary=self.sess.run([self.minimize,self.summary_op],{self.data:inp,self.decoder_input:dec,self.target:out})
          self.main_writer.add_summary(summary,global_step=i*no_of_batches+j)
        else:
          _=self.sess.run(self.minimize,{self.data:inp,self.decoder_input:dec,self.target:out})
        
      print('Epoch {} - {}s'.format(i,time.time()-time_init))
      if not self.seq2seq and record_interval is not None and i%record_interval==0:
        for writer,location in zip(self.auxiliary_writers,self.auxiliary_locations):
          write_interval_summary(writer,acquire_eval_lists([train_input,train_dec_input,train_output],location),i)
      
    if record_interval is not None:
      self.epoch=epoch #for range purpose, record total epoch
      self.record_interval=record_interval #for range purpose, record interval
       
    #def predict(self, sequence):