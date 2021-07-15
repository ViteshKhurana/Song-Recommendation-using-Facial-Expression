from django import forms
from .models import UserInput


class ImageForm(forms.ModelForm):
    class Meta:
        model = UserInput
        fields = ['image']
