from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse

from .models import Question, Essay, Score
from .forms import AnswerForm

from keras.models import load_model
import pickle
from .utils.pipeline import *
from .utils.customLayers import Conv1DWithMasking, Temporal_Mean_Pooling

import os
current_path = os.path.abspath(os.path.dirname(__file__))

# Load Tokenizers
tkns = {}
for prompt in range(1,9):
    with open(os.path.join(current_path, 'models/tokenizers/t%s.pickle' % prompt), 'rb') as handle:
        tkns[str(prompt)] = pickle.load(handle)
print("Loaded Tokenizers")

# Load Neural Model
model_name = ['LSTM_CNN', 'LSTM', 'RNN_CNN', 'RNN', 'GRU_CNN', 'GRU']
all_models = {}
for prompt in range(1,9):
    pred_models = {}
    for model in model_name:
        custom = {
            'Conv1DWithMasking' : Conv1DWithMasking,
            'Temporal_Mean_Pooling' : Temporal_Mean_Pooling
        }
        pred_models[model] = load_model(os.path.join(current_path, 'models/draft/%s_%s_model.h5' % (model, prompt)), custom_objects=custom)
        pred_models[model].load_weights(os.path.join(current_path, 'models/draft/%s_%s_weights.h5' % (model, prompt)))
    all_models[prompt] = pred_models
    print("Loaded Models for Prompt %s" % prompt)

# Main handler
def index(request, question_id=1):
    questions_list = Question.objects.order_by('set')
    question = get_object_or_404(Question, pk=question_id)
    form = ''

    if request.method == 'POST':    
        # create a form instance and populate it with data from the request:
        form = AnswerForm(request.POST)
        if form.is_valid():
            content = form.cleaned_data.get('answer')
            preds = {}
            pred_models = all_models[question_id]
            for model in pred_models:
                prompt = str(question_id)
                preds[model] = predict_score(prompt, pred_models[model], content, tkns)
          
            # Convert Predictions to Score Object
            model_list = []
            for x in preds:
                model_preds = Score.objects.create(
                    model_name = x,
                    model_pred = preds[x]
                )
                model_list.append(model_preds)
            
            # Store Essay/Prediction in Server
            Essay.objects.create(
                content=content,
                question=question
                # score_container=pred_container
            )
            context = {
                'questions_list': questions_list,
                "question": question,
                "form": form,
                "score": model_list
            }
            return render(request, 'index.html', context)
    else:
        form = AnswerForm()

    context = {
        'questions_list': questions_list,
        "question": question,
        "form": form,
    }
    return render(request, 'index.html', context)