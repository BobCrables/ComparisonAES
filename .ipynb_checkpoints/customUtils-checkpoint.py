import numpy as np
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta, Adamax
from keras.layers import Embedding

def get_optimizer(opt):

    clipvalue = 0
    clipnorm = 10

    if opt == 'rmsprop':
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
    elif opt == 'sgd':
        optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipnorm=clipnorm, clipvalue=clipvalue)
    elif opt == 'adagrad':
        optimizer = Adagrad(lr=0.01, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
    elif opt == 'adadelta':
        optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
    elif opt == 'adam':
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)
    elif opt == 'adamax':
        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)
    
    return optimizer

def limits(essay_type):
    """
    Returns the maximum/minimum scores for a essay set
    """
    data_dir = "data/training_set.tsv"
    originals = []
    
    fp=open(data_dir,'r', encoding="ascii", errors="ignore")
    fp.readline()
    for line in fp:
        temp=line.split("\t")
        if(temp[1]==essay_type): ## why only 4 ?? - evals in prompt specific fashion
            originals.append(float(temp[6]))
    fp.close()

    print("Min Score:", min(originals) , "| Max Score:", max(originals))
    range_min = min(originals)
    range_max = max(originals)
    return range_max, range_min

def preprocess_asap(essay_type,VALIDATION_SPLIT,TEST_SPLIT, glove_dir, data_dir):
    """
    Accepts an essay prompt and returns train, test and validation set embeddings.
    """
    
    EMBEDDING_DIM=300
    MAX_NB_WORDS=4000
    MAX_SEQUENCE_LENGTH=500
    DELTA=20
    
    texts=[]
    labels=[]
    sentences=[]
    originals = []
    
    print("Processing GloVe embedding...")
    
    fp1=open(glove_dir,"r", encoding="utf8")
    glove_emb={}
    for line in fp1:
        temp=line.split(" ")
        glove_emb[temp[0]]=np.asarray([float(i) for i in temp[1:]])

    print("Embedding done!")
    
    range_max, range_min = limits(essay_type)
    
    fp=open(data_dir,'r', encoding="ascii", errors="ignore")
    fp.readline()
    for line in fp:
        temp=line.split("\t")
        if(temp[1]==essay_type): ## why only 4 ?? - evals in prompt specific fashion
            originals.append(float(temp[6]))
    fp.close()

    fp=open(data_dir,'r', encoding="ascii", errors="ignore")
    fp.readline()
    sentences=[]
    for line in fp:
        temp=line.split("\t")
        if(temp[1]==essay_type):
            texts.append(temp[2])
            labels.append((float(temp[6])-range_min)/(range_max-range_min))
            line=temp[2].strip()
            sentences.append(nltk.tokenize.word_tokenize(line))

    fp.close()
    
    print("Number of Training Essays: %s" %len(texts))
    labels=np.asarray(labels)
    
    for i in sentences:
        temp1=np.zeros((1, EMBEDDING_DIM))
        for w in i:
            if(w in glove_emb):
                temp1+=glove_emb[w]
        temp1/=len(i)
    
    tokenizer=Tokenizer() #num_words=MAX_NB_WORDS) #limits vocabulory size
    tokenizer.fit_on_texts(texts)
    sequences=tokenizer.texts_to_sequences(texts) #returns list of sequences
    word_index=tokenizer.word_index #dictionary mapping
    print('Found %s unique tokens!' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    
    embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM))
    
    for word,i in word_index.items():
        if(i>=len(word_index)):
            continue
        if word in glove_emb:
                embedding_matrix[i]=glove_emb[word]
    vocab_size=len(word_index)
    
    embedding_layer=Embedding(vocab_size,EMBEDDING_DIM,weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                mask_zero=True,
                                trainable=False)
    side_embedding_layer=Embedding(vocab_size,EMBEDDING_DIM,weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    mask_zero=False,
                                    trainable=False)
    
    # Split train & test data
    print('Splitting training/test data...')
    indices=np.arange(data.shape[0])
    np.random.shuffle(indices)
    data=data[indices]
    labels=labels[indices]
    validation_size=int(VALIDATION_SPLIT*data.shape[0])
    
    x_train=data[:-validation_size]
    y_train=labels[:-validation_size]
    x_notrain=data[-validation_size:]
    y_notrain=labels[-validation_size:]

    test_size=int(TEST_SPLIT*x_notrain.shape[0])
    x_val=x_notrain[:-test_size]
    y_val=y_notrain[:-test_size]
    x_test=x_notrain[-test_size:]
    y_test=y_notrain[-test_size:]
    print('Done.')
    
    return x_train, y_train, x_val, y_val, x_test, y_test, embedding_layer