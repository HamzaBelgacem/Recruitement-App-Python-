from django.shortcuts import render, redirect, get_object_or_404
from .models import Profile, Skill, AppliedJobs, SavedJobs
from recruiters.models import Job, Applicants, Selected
from .forms import ProfileUpdateForm, NewSkillForm
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.http import HttpResponseRedirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.views.generic import UpdateView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import get_user_model
from django.core.paginator import Paginator



from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
import docx2txt
import PyPDF2
import os
from os import listdir
from os.path import isfile, join
from io import StringIO
import pandas as pd
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher
import csv

from candidates.models import Profile, Skill 
from recruiters.models import Job 




import numpy as np
# import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten #couche d'applatissement
from keras.layers.convolutional import Conv2D,MaxPooling2D #conv2D:couche de convolution,MzxPooling2D:couche de mise en commun
import pickle #save model


def home(request):
    context = {
        'home_page': "active",
    }
    return render(request, 'candidates/home.html', context)


def job_search_list(request):
    query = request.GET.get('p')
    loc = request.GET.get('q')
    object_list = []
    if(query == None):
        object_list = Job.objects.all()
    else:
        title_list = Job.objects.filter(
            title__icontains=query).order_by('-date_posted')
        skill_list = Job.objects.filter(
            skills_req__icontains=query).order_by('-date_posted')
        company_list = Job.objects.filter(
            company__icontains=query).order_by('-date_posted')
        job_type_list = Job.objects.filter(
            job_type__icontains=query).order_by('-date_posted')
        for i in title_list:
            object_list.append(i)
        for i in skill_list:
            if i not in object_list:
                object_list.append(i)
        for i in company_list:
            if i not in object_list:
                object_list.append(i)
        for i in job_type_list:
            if i not in object_list:
                object_list.append(i)
    if(loc == None):
        locat = Job.objects.all()
    else:
        locat = Job.objects.filter(
            location__icontains=loc).order_by('-date_posted')
    final_list = []
    for i in object_list:
        if i in locat:
            final_list.append(i)
    paginator = Paginator(final_list, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {
        'jobs': page_obj,
        'query': query,
    }
    return render(request, 'candidates/job_search_list.html', context)


def job_detail(request, slug):
    job = get_object_or_404(Job, slug=slug)
    apply_button = 0
    save_button = 0
    profile = Profile.objects.filter(user=request.user).first()
    if AppliedJobs.objects.filter(user=request.user).filter(job=job).exists():
        apply_button = 1
    if SavedJobs.objects.filter(user=request.user).filter(job=job).exists():
        save_button = 1
    relevant_jobs = []
    jobs1 = Job.objects.filter(
        company__icontains=job.company).order_by('-date_posted')
    jobs2 = Job.objects.filter(
        job_type__icontains=job.job_type).order_by('-date_posted')
    jobs3 = Job.objects.filter(
        title__icontains=job.title).order_by('-date_posted')
    for i in jobs1:
        if len(relevant_jobs) > 5:
            break
        if i not in relevant_jobs and i != job:
            relevant_jobs.append(i)
    for i in jobs2:
        if len(relevant_jobs) > 5:
            break
        if i not in relevant_jobs and i != job:
            relevant_jobs.append(i)
    for i in jobs3:
        if len(relevant_jobs) > 5:
            break
        if i not in relevant_jobs and i != job:
            relevant_jobs.append(i)

    return render(request, 'candidates/job_detail.html', {'job': job, 'profile': profile, 'apply_button': apply_button, 'save_button': save_button, 'relevant_jobs': relevant_jobs, 'candidate_navbar': 1})


@login_required
def saved_jobs(request):
    jobs = SavedJobs.objects.filter(
        user=request.user).order_by('-date_posted')
    return render(request, 'candidates/saved_jobs.html', {'jobs': jobs, 'candidate_navbar': 1})


@login_required
def applied_jobs(request):
    jobs = AppliedJobs.objects.filter(
        user=request.user).order_by('-date_posted')
    statuses = []
    for job in jobs:
        if Selected.objects.filter(job=job.job).filter(applicant=request.user).exists():
            statuses.append(0)
        elif Applicants.objects.filter(job=job.job).filter(applicant=request.user).exists():
            statuses.append(1)
        else:
            statuses.append(2)
    zipped = zip(jobs, statuses)
    return render(request, 'candidates/applied_jobs.html', {'zipped': zipped, 'candidate_navbar': 1})


@login_required
def intelligent_search(request):
    relevant_jobs = []
    common = []
    job_skills = []
    user = request.user
    profile = Profile.objects.filter(user=user).first()
    my_skill_query = Skill.objects.filter(user=user)
    my_skills = []
    for i in my_skill_query:
        my_skills.append(i.skill.lower())
    if profile:
        jobs = Job.objects.filter(
            job_type=profile.looking_for).order_by('-date_posted')
    else:
        jobs = Job.objects.all()
    for job in jobs:
        skills = []
        sk = str(job.skills_req).split(",")
        for i in sk:
            skills.append(i.strip().lower())
        common_skills = list(set(my_skills) & set(skills))
        if (len(common_skills) != 0 and len(common_skills) >= len(skills)//2):
            relevant_jobs.append(job)
            common.append(len(common_skills))
            job_skills.append(len(skills))
    objects = zip(relevant_jobs, common, job_skills)
    objects = sorted(objects, key=lambda t: t[1]/t[2], reverse=True)
    objects = objects[:100]
    context = {
        'intel_page': "active",
        'jobs': objects,
        'counter': len(relevant_jobs),
    }
    return render(request, 'candidates/intelligent_search.html', context)


@login_required
def my_profile(request):
    you = request.user
    profile = Profile.objects.filter(user=you).first()
    user_skills = Skill.objects.filter(user=you)
    if request.method == 'POST':
        form = NewSkillForm(request.POST)
        if form.is_valid():
            data = form.save(commit=False)
            data.user = you
            data.save()
            return redirect('my-profile')
    else:
        form = NewSkillForm()
    context = {
        'u': you,
        'profile': profile,
        'skills': user_skills,
        'form': form,
        'profile_page': "active",
    }
    return render(request, 'candidates/profile.html', context)


@login_required
def edit_profile(request):
    you = request.user
    profile = Profile.objects.filter(user=you).first()
    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            data = form.save(commit=False)
            data.user = you
            data.save()
            return redirect('my-profile')
    else:
        form = ProfileUpdateForm(instance=profile)
    context = {
        'form': form,
    }
    return render(request, 'candidates/edit_profile.html', context)


@login_required
def profile_view(request, slug):
    p = Profile.objects.filter(slug=slug).first()
    you = p.user
    user_skills = Skill.objects.filter(user=you)
    context = {
        'u': you,
        'profile': p,
        'skills': user_skills,
    }
    return render(request, 'candidates/profile.html', context)


def candidate_details(request):
    return render(request, 'candidates/details.html')


@login_required
@csrf_exempt
def delete_skill(request, pk=None):
    if request.method == 'POST':
        id_list = request.POST.getlist('choices')
        for skill_id in id_list:
            Skill.objects.get(id=skill_id).delete()
        return redirect('my-profile')


@login_required
def save_job(request, slug):
    user = request.user
    job = get_object_or_404(Job, slug=slug)
    saved, created = SavedJobs.objects.get_or_create(job=job, user=user)
    return HttpResponseRedirect('/job/{}'.format(job.slug))


@login_required
def apply_job(request, slug):
    user = request.user
    job = get_object_or_404(Job, slug=slug)
    applied, created = AppliedJobs.objects.get_or_create(job=job, user=user)
    applicant, creation = Applicants.objects.get_or_create(
        job=job, applicant=user)
    return HttpResponseRedirect('/job/{}'.format(job.slug))


@login_required
def remove_job(request, slug):
    user = request.user
    job = get_object_or_404(Job, slug=slug)
    saved_job = SavedJobs.objects.filter(job=job, user=user).first()
    saved_job.delete()
    return HttpResponseRedirect('/job/{}'.format(job.slug))




def pdfextract(file):
    fileReader = PyPDF2.PdfFileReader(open(file,'rb'))
    countpage = fileReader.getNumPages()
    count = 0
    text = []
    while count < countpage:    
        pageObj = fileReader.getPage(count)
        count +=1
        t = pageObj.extractText()
        #print (t)
        text.append(t)
    return text




from candidates.models import SavedJobs


# def skill_cand(request):
#     #Statistics=["Statical models","Statical modeling", "probabilty", "normal distribution", "factor analysis", "markov chain" ,"monte carlo"]
#     #Machine_Learning=["linear regression","logistic regression","k means","random forest","svm","pca","decision trees","svd","ensembles models", "boltzman machine","naive bayes","supporor machinet vect"]
#     current_user=request.user.id
#     skills_req = SavedJobs.objects.filter(user_id=current_user)
#     #print(skills_req)
#     total_jobs=[]
#     for i in skills_req:
#         with open('names_'+str(i)+'.csv', 'w', newline='') as csvfile:
#             fieldnames = ['skills']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             #for i in skills_req:
#             l=(i.job.skills_req).split(',')
#             for j in l:
#                 writer.writerow({'skills': j})
#             total_jobs.append(('names_'+str(i)+'.csv'))
#     print("totalll",total_jobs)
    
#     def create_profile(file):
#         text = pdfextract(file) 
#         text = str(text)
#         text = text.replace("\\n", "")
#         text = text.lower()
#         #below is the csv where we have all the keywords, you can customize your own
#         for i in total_jobs:
#             keyword_dict = pd.read_csv(i,sep=",")
#             stats_words = [nlp(text) for text in keyword_dict['skills'].dropna(axis = 0)]
#             matcher = PhraseMatcher(nlp.vocab)
#             matcher.add('Stats', None, *stats_words)
#             doc = nlp(text)
        
#             d = []  
#             matches = matcher(doc)
#             for match_id, start, end in matches:
#                 rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
#                 span = doc[start : end]  # get the matched slice of the doc
#                 d.append((rule_id, span.text))      
#             keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())
        
#             ## convertimg string of keywords to dataframe
#             df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
#             df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
#             df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
#             df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
#             df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
        
#             base = os.path.basename(file)
#             filename = os.path.splitext(base)[0]
        
#             name = filename.split('_')
#             name2 = name[0]
#             name2 = name2.lower()
#             ## converting str to dataframe
#             name3 = pd.read_csv(StringIO(name2),names = ['Candidate Name'])
            
#             dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
#             dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace = True)

#         return(dataf)

#     final_database=pd.DataFrame()
#     i = 0 
#     current_user=request.user.id
#     data=Profile.objects.get(user_id=current_user)
#     x=str(data.resume)
#     f=x.replace("/","\\")
#     onlyfiles=[]
#     resume="C:/Users/hamza/Desktop/recruitement app/media/"+str(f)#.split("/")[1]
#     onlyfiles.append(resume)
        
#     # mypath='C:/Users/hp/Desktop/Fenice-Network2/Fenice-Network/media/resumes' #enter your path here where you saved the resumes
#     # onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

#     # print("oooooooooo",onlyfiles)

#     while i < len(onlyfiles):
#         file = onlyfiles[i]
#         dat = create_profile(file)
#         final_database = final_database.append(dat)
#         i +=1

#     final_database2 = final_database['Keyword'].groupby([final_database['Candidate Name'], final_database['Subject']]).count().unstack()
#     final_database2.reset_index(inplace = True)
#     final_database2.fillna(0,inplace=True)
#     new_data = final_database2.iloc[:,1:]
#     new_data.index = final_database2['Candidate Name']
#     print("newwwwwwwwwwwww",new_data.index)

#     liste=[]
#     for i,j in new_data.iterrows():
#         liste.append(i)

#     x=new_data.values
#     skills=x.tolist()
#     liste_finale=list(zip(liste,skills))
#     print("le5ra",liste_finale)

#     liste_f=list()
#     for i,j in liste_finale:
#         liste_f.append((i,sum(j)))
#     print("liste_f",liste_f)
#     for i in total_jobs:
#         feature = pd.read_csv(i, sep=",")
#         #print("cccccccccccc",feature)
#         liste_stat=[]
#         indices=["skills"]
#         liste2=[]
#         for i in indices:
#             feature[i]
#         print("featuures",feature["skills"].values)
#         #x=liste.tolist()
#         #liste2.append(x)
#         #print("liste2",liste2)
#         liste3=feature["skills"].values
#     # for j in liste2:
#     #     for y in j:
#     #         if type(y)!=float:
#     #             liste3.append(y)
#         total=len(liste3)
#         liste=[]
#     # print(liste_f)
#     # for i,j in liste_f:
#     #     liste.append(( str(i)+"'s resume matches about "+ str((float(int(j)/len(liste3))*100))+ "% of the job description."))

        
#     context={"feature":feature,"skills_req":skills_req}
#     #return render(request,"recruiters/score.html",context)

#     return render(request,"candidates/score.html", context)

    

def create_profile(file):
    text = pdfextract(file) 
    text = str(text)
    text = text.replace("\\n", "")
    text = text.lower()
    #below is the csv where we have all the keywords, you can customize your own
    keyword_dict = pd.read_csv('C:/Users/hamza/Desktop/recruitement app/recruiters/template_new.csv',sep=";")
    stats_words = [nlp(text) for text in keyword_dict['Statistics'].dropna(axis = 0)]
    NLP_words = [nlp(text) for text in keyword_dict['NLP'].dropna(axis = 0)]
    ML_words = [nlp(text) for text in keyword_dict['Machine Learning'].dropna(axis = 0)]
    DL_words = [nlp(text) for text in keyword_dict['Deep Learning'].dropna(axis = 0)]
    R_words = [nlp(text) for text in keyword_dict['R Language'].dropna(axis = 0)]
    python_words = [nlp(text) for text in keyword_dict['Python Language'].dropna(axis = 0)]
    Data_Engineering_words = [nlp(text) for text in keyword_dict['Data Engineering'].dropna(axis = 0)]

    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('Stats', None, *stats_words)
    matcher.add('NLP', None, *NLP_words)
    matcher.add('ML', None, *ML_words)
    matcher.add('DL', None, *DL_words)
    matcher.add('R', None, *R_words)
    matcher.add('Python', None, *python_words)
    matcher.add('DE', None, *Data_Engineering_words)
    doc = nlp(text)
    
    d = []  
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start : end]  # get the matched slice of the doc
        d.append((rule_id, span.text))      
    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())
    
    ## convertimg string of keywords to dataframe
    df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
    
    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]
       
    name = filename.split('_')
    name2 = name[0]
    name2 = name2.lower()
    ## converting str to dataframe
    name3 = pd.read_csv(StringIO(name2),names = ['Candidate Name'])
    
    dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
    dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace = True)

    return(dataf)





def skill_cand(request):
    final_database=pd.DataFrame()
    i = 0 
    current_user=request.user.id
    data=Profile.objects.get(user_id=current_user)
    print("data",data)
    print("resume",data.resume)
    print(type(data.resume))
    print(str(data.resume))
    print(data.resume.url)
    print (type(data.resume.url))
    x=str(data.resume)
    f=x.replace("/","\\")
    onlyfiles=[]
    resume="C:/Users/hamza/Desktop/recruitement app/media/"+str(f)#.split("/")[1]
    onlyfiles.append(resume)
    print("oooooooooo",onlyfiles)
        
    # mypath='C:/Users/hp/Desktop/Fenice-Network2/Fenice-Network/media/resumes' #enter your path here where you saved the resumes
    # onlyfiles = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

    # print("oooooooooo",onlyfiles)

    while i < len(onlyfiles):
        file = onlyfiles[i]
        dat = create_profile(file)
        final_database = final_database.append(dat)
        i +=1

    final_database2 = final_database['Keyword'].groupby([final_database['Candidate Name'], final_database['Subject']]).count().unstack()
    final_database2.reset_index(inplace = True)
    final_database2.fillna(0,inplace=True)
    new_data = final_database2.iloc[:,1:]
    new_data.index = final_database2['Candidate Name']
    liste=[]
    for i,j in new_data.iterrows():
        liste.append(i)

    x=new_data.values
    skills=x.tolist()
    liste_finale=list(zip(liste,skills))

    liste_f=list()
    for i,j in liste_finale:
        liste_f.append((i,sum(j)))

    feature = pd.read_csv("C:/Users/hamza/Desktop/recruitement app/recruiters/template_new.csv", sep=";")
    liste_stat=[]
    indices=["Statistics","Machine Learning","Deep Learning","R Language","Python Language","NLP","Data Engineering"]
    liste2=[]
    for i in indices:
        feature[i]
        liste=feature[i].values
        #liste_stat.append(feature["Statistics"].values)
        x=liste.tolist()
        liste2.append(x)
    liste3=[]
    for j in liste2:
        for y in j:
            if type(y)!=float:
                liste3.append(y)
    total=len(liste3)
    liste=[]
    print(liste_f)
    for i,j in liste_f:
        liste.append(( str(i)+"'s resume matches about "+ str((float(int(j)/len(liste3))*100))+ "% of the job description."))

        
    context={"liste":liste,"data":data}
    return render(request,"candidates/score.html",context)




import numpy as np
#import cv2
#import pickle
from keras.models import load_model

from pdf2image import convert_from_path
#import urllib.request
# SETUP THE VIDEO CAMERA
#cap = cv2.VideoCapture(0)
#model=load_model("C:/Users/Hamza/Desktop/recruitement app/candidates/model.h5")

#font=cv2.FONT_HERSHEY_SIMPLEX
#url='http://192.168.43.1:8080/shot.jpg'
# IMPORT THE TRANNIED MODEL
def grayscale(img):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    #img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if   classNo == 0: return 'Organised CV'
    elif classNo == 1: return 'not an Organised CV'



def organiser(request):
    current_user=request.user.id
    data=Profile.objects.get(user_id=current_user)
    print("data",data)
    print("resume",data.resume)
    print(type(data.resume))
    print(str(data.resume))
    images = convert_from_path("C:/Users/hamza/Desktop/recruitement app"+str(data.resume.url), 500)
    for i, image in enumerate(images):
        fname = 'image'+str(i)+'.png'
        image.save(fname, "PNG")
    
    
    #imgOrignal = cv2.imread("C:/Users/hamza/Desktop/recruitement app/image0.png")

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    #img = cv2.resize(img, (300,300))
    img = preprocessing(img)
    #cv2.imshow("Processed Image", img)
    img = img.reshape(1, 300, 300, 1)
    
   #cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    # predictions = model.predict(img)
    # classIndex = model.predict_classes(img)
    # probabilityValue =np.amax(predictions)
    # ok=False
    # liste=[]
    # test="CV organise"
    # if probabilityValue > 0.5:     
    #     if getCalssName(classIndex)=="not an Organised CV":
    #         liste.append(getCalssName(classIndex))
    #         ok=True
    #         print("Not Organized CV")
    #     if getCalssName(classIndex)=="Organised CV":
    #         print("Organized CV")      

    # context={"data":data,"predictions":predictions,"classIndex": getCalssName(classIndex),"ok":ok,"liste":liste,"test":test}
    return render(request,"candidates/score2.html",context)

    