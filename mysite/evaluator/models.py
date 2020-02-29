from django.db import models

class Question(models.Model):
    """ A model of the 8 questions. """
    question_title = models.TextField(max_length=100000)
    set = models.IntegerField(unique=True)
    min_score = models.IntegerField()
    max_score = models.IntegerField()
    question_desc = models.TextField(max_length=100000)
    grade_level = models.IntegerField()
    question_type = models.TextField(max_length=100000)
    training_data_size = models.IntegerField()
    essay_length = models.IntegerField()
    high_sample = models.TextField(max_length=100000)
    med_sample = models.TextField(max_length=100000)
    low_sample = models.TextField(max_length=100000)

    def __str__(self):
        return str(self.set)

class Score(models.Model):
    """ A model of the training model responses """
    model_name = models.CharField(max_length=50, db_index=True)
    model_pred = models.IntegerField()

    # def __str__(self):
    #     return str(self.set)

class Essay(models.Model):
    """ Essay to be submitted. """
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    content = models.TextField(max_length=100000)
    score_container = models.ForeignKey(Score, on_delete=models.CASCADE)

    # score = models.ForeignKey(Score, on_delete=models.CASCADE)
    # score = models.IntegerField(null=True, blank=True)