from .form import ImageForm
from django.shortcuts import render
import numpy as np
from .functions import resize_image, loadModel
import cv2
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def index(request):
    if request.method == 'POST':
        form = ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            obj = form.instance
            model = loadModel('./facialExpression/CNN_HDF5_Format.h5')
            img = cv2.imread(f'.{obj.image.url}', 0)
            resized_image = resize_image(img)
            result = np.argmax(model.predict(resized_image), axis=1)
            result = labels[result[0]]
            param = {'obj': obj, 'facialExpression': result}
            return render(request, 'facialExpression/index2.html', param)
    else:
        form = ImageForm()
    return render(request, 'facialExpression/index2.html', {'form': form})


def result_page(request):
    return render(request, 'facialExpression/result.html')