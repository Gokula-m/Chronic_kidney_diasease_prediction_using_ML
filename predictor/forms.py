from django import forms


YES_NO_CHOICES = [
    ("", "Unknown"),
    ("yes", "Yes"),
    ("no", "No"),
]
NORMAL_CHOICES = [
    ("", "Unknown"),
    ("normal", "Normal"),
    ("abnormal", "Abnormal"),
]
PRESENT_CHOICES = [
    ("", "Unknown"),
    ("present", "Present"),
    ("notpresent", "Not present"),
]
APPETITE_CHOICES = [
    ("", "Unknown"),
    ("good", "Good"),
    ("poor", "Poor"),
]


class CKDPredictionForm(forms.Form):
    age = forms.FloatField(label="Age", required=False, min_value=0)
    bp = forms.FloatField(label="Blood pressure", required=False, min_value=0)
    sg = forms.FloatField(label="Specific gravity", required=False)
    al = forms.FloatField(label="Albumin", required=False, min_value=0)
    su = forms.FloatField(label="Sugar", required=False, min_value=0)
    bgr = forms.FloatField(label="Blood glucose random", required=False, min_value=0)
    bu = forms.FloatField(label="Blood urea", required=False, min_value=0)
    sc = forms.FloatField(label="Serum creatinine", required=False, min_value=0)
    sod = forms.FloatField(label="Sodium", required=False, min_value=0)
    pot = forms.FloatField(label="Potassium", required=False, min_value=0)
    hemo = forms.FloatField(label="Hemoglobin", required=False, min_value=0)
    pcv = forms.FloatField(label="Packed cell volume", required=False, min_value=0)
    wc = forms.FloatField(label="White blood cell count", required=False, min_value=0)
    rc = forms.FloatField(label="Red blood cell count", required=False, min_value=0)

    rbc = forms.ChoiceField(label="Red blood cells", required=False, choices=NORMAL_CHOICES)
    pc = forms.ChoiceField(label="Pus cell", required=False, choices=NORMAL_CHOICES)
    pcc = forms.ChoiceField(label="Pus cell clumps", required=False, choices=PRESENT_CHOICES)
    ba = forms.ChoiceField(label="Bacteria", required=False, choices=PRESENT_CHOICES)
    htn = forms.ChoiceField(label="Hypertension", required=False, choices=YES_NO_CHOICES)
    dm = forms.ChoiceField(label="Diabetes mellitus", required=False, choices=YES_NO_CHOICES)
    cad = forms.ChoiceField(label="Coronary artery disease", required=False, choices=YES_NO_CHOICES)
    appet = forms.ChoiceField(label="Appetite", required=False, choices=APPETITE_CHOICES)
    pe = forms.ChoiceField(label="Pedal edema", required=False, choices=YES_NO_CHOICES)
    ane = forms.ChoiceField(label="Anemia", required=False, choices=YES_NO_CHOICES)

    def as_model_input(self) -> dict:
        return {key: (None if value == "" else value) for key, value in self.cleaned_data.items()}
