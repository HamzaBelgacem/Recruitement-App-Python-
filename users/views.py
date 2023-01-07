from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages 
from django.contrib.auth import authenticate, login as dj_login, logout
from candidates.models import Profile
from users.forms import SignUpForm 

def login(request):
    return render(request, 'users/index2.html')



def login_form(request):
    if request.method=="POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            dj_login(request,user)
            # current_user=request.user
            # print(current_user)
        #     #userprofile=Profile.objects.get(user_id=current_user.id)
        #     #request.session["userimage"]=userprofile.image.url
        #     # Redirect to a success page.
        return HttpResponseRedirect("/")
           
        # else:
        #     # Return an 'invalid login' error message.
        #     messages.warning(request,"Login Error ! Username or Password is incorrect ")
        #     return HttpResponseRedirect("/login")

    return render(request,"users/index2.html",{})



@login_required
def account(request):
    context = {
        'account_page': "active",
    }
    return render(request, 'users/account.html', context)


def privacy(request):
    return render(request, 'users/privacy.html')


def terms(request):
    return render(request, 'users/terms.html')


def pricing(request):
    context = {
        'rec_navbar': 1,
    }
    return render(request, 'users/pricing.html', context)



def register_form(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save() #completed sign up
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            dj_login(request, user)
            return HttpResponseRedirect('/')
        # else:
        #     messages.warning(request, form.errors)
        #     return HttpResponseRedirect('/register')

    form = SignUpForm()
    context = {'form': form}

    return render(request, "users/register.html", context)