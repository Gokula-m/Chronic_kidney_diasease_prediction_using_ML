from django.shortcuts import render

from .forms import CKDPredictionForm
from .ml import predict_ckd


def home(request):
    result = None

    if request.method == "POST":
        form = CKDPredictionForm(request.POST)
        if form.is_valid():
            result = predict_ckd(form.as_model_input())
    else:
        form = CKDPredictionForm()

    return render(
        request,
        "predictor/home.html",
        {
            "form": form,
            "result": result,
        },
    )

# Create your views here.
