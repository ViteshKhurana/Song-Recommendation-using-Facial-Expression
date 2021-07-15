from django.http import HttpResponse
from .models import UserInput
from .form import ImageForm
from django.shortcuts import render,redirect
import numpy as np
from django.conf import settings as django_settings
import os
from django.core.files.storage import FileSystemStorage
from .functions import resize_image, loadModel
import cv2
import os
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def index(request):
    return render(request,'facialExpression/index.html')


def emotionPrediction(request):
    if request.method == 'POST':
        form=ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            obj=form.instance
            model=loadModel('./facialExpression/CNN_HDF5_Format.h5')
            img = cv2.imread(obj.image, 0)
            resized_image = resize_image(img)
            result = np.argmax(model.predict(resized_image), axis=1)
            result = labels[result[0]]
            param={'obj':obj,'facialExpression':result}
            return render(request,'facialExpression/index.html',param)
    return HttpResponse("Try Again")


def resultPage(request):
    return render(request,'facialExpression/result.html')