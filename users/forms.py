from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.contrib.auth.models import User

from django.forms import TextInput, EmailInput, Select, FileInput
from django.utils.translation import gettext_lazy as _

class SignUpForm(UserCreationForm):
    username = forms.CharField(max_length=30,label= _('User Name :'))
    email = forms.EmailField(max_length=200,label= _('Email :'))
    first_name = forms.CharField(max_length=100, help_text=_('First Name'),label= _('First Name :'))
    last_name = forms.CharField(max_length=100, help_text=_('Last Name'),label= _('Last Name :'))

    class Meta:
        model = User
        fields = ('username', 'email','first_name','last_name', 'password1', 'password2', )