import warnings
warnings.filterwarnings("ignore")
import nltk
import pickle
from keras.models import load_model
from pipeline import get_tokenizer, predict_score

# Load Tokenizers
tkns = {}
for prompt in range(1,9):
    with open('models/tokenizers/t%s.pickle' % prompt, 'rb') as handle:
        tkns[str(prompt)] = pickle.load(handle)
        
# Load Neural Model
prompt = '1'
model_name = 'LSTM_CNN'

from customLayers import Conv1DWithMasking, Temporal_Mean_Pooling
custom = {
    'Conv1DWithMasking' : Conv1DWithMasking,
    'Temporal_Mean_Pooling' : Temporal_Mean_Pooling
}
model = load_model('models/draft/%s_%s_model.h5' % (model_name, prompt), custom_objects=custom)
model.load_weights('models/draft/%s_%s_weights.h5' % (model_name, prompt))

print('Loaded Prompt: %s; Model: %s' % (prompt, model_name))

essay = input("Type an essay: ") 
score = predict_score(prompt, model, essay, tkns)

print(score)

pause = input("Pause") 