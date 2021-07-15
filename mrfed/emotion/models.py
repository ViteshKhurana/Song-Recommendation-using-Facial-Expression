from django.db import models


class UserInput(models.Model):
    image = models.ImageField(upload_to='facialExpression/images/')

    def __str__(self):
        return "successful"
