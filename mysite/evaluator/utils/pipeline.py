import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
import numpy as np

# Import custom Keras layers
from . import customLayers

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 500

def score_range(prompt):
    range = {"1": (2, 12), "2": (1, 6), "3": (0, 3), "4": (0, 3), "5": (0, 4), "6": (0, 4), "7": (0, 30), "8": (0, 60)}
    return range[prompt][0], range[prompt][1]

def glove_emb(glove_dir):
    print("Processing GloVe embedding")
    fp1=open(glove_dir,"r", encoding="utf8")
    glove_emb={}
    for line in fp1:
        temp=line.split(" ")
        glove_emb[temp[0]]=np.asarray([float(i) for i in temp[1:]])
    return glove_emb

def extract_essays(prompt, data_dir, glove_emb):
    texts=[]
    labels=[]
    sentences=[]
    range_min, range_max = score_range(prompt)
    
    fp=open(data_dir,'r', encoding="ascii", errors="ignore")
    fp.readline()
    for line in fp:
        temp=line.split("\t")
        if(temp[1]==prompt):
            texts.append(temp[2])
            labels.append((float(temp[6])-range_min)/(range_max-range_min))
            line=temp[2].strip()
            sentences.append(nltk.tokenize.word_tokenize(line))
    fp.close()
    labels=np.asarray(labels)
    for i in sentences:
        temp1=np.zeros((1, EMBEDDING_DIM))
        for w in i:
            if(w in glove_emb):
                temp1+=glove_emb[w]
        temp1/=len(i)
    return texts

def embed_layer(word_index, glove_emb):
    embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM))
    
    for word,i in word_index.items():
        if(i>=len(word_index)):
            continue
        if word in glove_emb:
                embedding_matrix[i]=glove_emb[word]
    vocab_size=len(word_index)
    
    embed_layer=Embedding(vocab_size,EMBEDDING_DIM,weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                mask_zero=True,
                                trainable=False)
    side_embed_layer=Embedding(vocab_size,EMBEDDING_DIM,weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    mask_zero=False,
                                    trainable=False)
    return embed_layer

def get_tokenizer(prompt, data_dir, glove_dir):
    EMBEDDING_DIM=300
    MAX_SEQUENCE_LENGTH=500
    DELTA=20
        
    range_min, range_max = score_range(prompt)
    embed = glove_emb(glove_dir)
    texts = extract_essays(prompt, data_dir, embed)
    
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences=tokenizer.texts_to_sequences(texts) # Returns list of sentences
    word_index=tokenizer.word_index # Maps all tokens to tokenizer

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return tokenizer

def vectorize(essay, prompt, tokenizer):
    
    temp = [essay, ""]
    
    sequenced_essay = tokenizer.texts_to_sequences(temp) # Returns list of sentences
    temp_vec = pad_sequences(sequenced_essay, maxlen=MAX_SEQUENCE_LENGTH)
    return temp_vec

def predict_score(prompt, model, essay, tkns):
    # Find Tokenizer
    tokener = tkns[prompt]
    
    # Generate Essay Vector
    essay_vec = vectorize(essay, prompt, tokener)
    
    # Predict and normalize score
    y_pred = model.predict([essay_vec])
    range_min, range_max = score_range(prompt)
    y_pred[1] = 0 # This is a very jank solution to a bug
    y_pred_fin =[int(round(a*(range_max-range_min)+range_min)) for a in y_pred.reshape(y_pred.shape[0]).tolist()]
    score = y_pred_fin[0]
    return score