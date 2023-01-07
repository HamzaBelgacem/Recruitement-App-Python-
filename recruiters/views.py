from django.shortcuts import render, redirect, get_object_or_404
from .models import Job, Applicants, Selected
from candidates.models import Profile, Skill
from .forms import NewJobForm
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.http import HttpResponseRedirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.views.generic import UpdateView
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.core.paginator import Paginator
from django.contrib.auth import authenticate



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

from candidates.models import Profile




def rec_details(request):
    context = {
        'rec_home_page': "active",
        'rec_navbar': 1,
    }
    return render(request, 'recruiters/details.html', context)


@login_required
def add_job(request):
    user = request.user
    if request.method == "POST":
        form = NewJobForm(request.POST)
        if form.is_valid():
            data = form.save(commit=False)
            data.recruiter = user
            data.save()
            return redirect('job-list')
    else:
        form = NewJobForm()
    context = {
        'add_job_page': "active",
        'form': form,
        'rec_navbar': 1,
    }
    return render(request, 'recruiters/add_job.html', context)


@login_required
def edit_job(request, slug):
    user = request.user
    job = get_object_or_404(Job, slug=slug)
    if request.method == "POST":
        form = NewJobForm(request.POST, instance=job)
        if form.is_valid():
            data = form.save(commit=False)
            data.save()
            return redirect('add-job-detail', slug)
    else:
        form = NewJobForm(instance=job)
    context = {
        'form': form,
        'rec_navbar': 1,
        'job': job,
    }
    return render(request, 'recruiters/edit_job.html', context)


@login_required
def job_detail(request, slug):
    job = get_object_or_404(Job, slug=slug)
    context = {
        'job': job,
        'rec_navbar': 1,
    }
    return render(request, 'recruiters/job_detail.html', context)


@login_required
def all_jobs(request):
    jobs = Job.objects.filter(recruiter=request.user).order_by('-date_posted')
    paginator = Paginator(jobs, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {
        'manage_jobs_page': "active",
        'jobs': page_obj,
        'rec_navbar': 1,
    }
    return render(request, 'recruiters/job_posts.html', context)


@login_required
def search_candidates(request):
    profile_list = Profile.objects.all()
    profiles = []
    for profile in profile_list:
        if profile.resume and profile.user != request.user:
            profiles.append(profile)

    rec1 = request.GET.get('r')
    rec2 = request.GET.get('s')

    if rec1 == None:
        li1 = Profile.objects.all()
    else:
        li1 = Profile.objects.filter(location__icontains=rec1)

    if rec2 == None:
        li2 = Profile.objects.all()
    else:
        li2 = Profile.objects.filter(looking_for__icontains=rec2)

    final = []
    profiles_final = []

    for i in li1:
        if i in li2:
            final.append(i)

    for i in final:
        if i in profiles:
            profiles_final.append(i)

    paginator = Paginator(profiles_final, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {
        'search_candidates_page': "active",
        'rec_navbar': 1,
        'profiles': page_obj,
    }
    return render(request, 'recruiters/candidate_search.html', context)


@login_required
def job_candidate_search(request, slug):
    job = get_object_or_404(Job, slug=slug)
    relevant_candidates = []
    common = []
    applicants = Profile.objects.filter(looking_for=job.job_type)
    job_skills = []
    skills = str(job.skills_req).split(",")
    for skill in skills:
        job_skills.append(skill.strip().lower())
    for applicant in applicants:
        user = applicant.user
        skill_list = list(Skill.objects.filter(user=user))
        skills = []
        for i in skill_list:
            skills.append(i.skill.lower())
        common_skills = list(set(job_skills) & set(skills))
        if (len(common_skills) != 0 and len(common_skills) >= len(job_skills)//2):
            relevant_candidates.append(applicant)
            common.append(len(common_skills))
    objects = zip(relevant_candidates, common)
    objects = sorted(objects, key=lambda t: t[1], reverse=True)
    objects = objects[:100]
    context = {
        'rec_navbar': 1,
        'job': job,
        'objects': objects,
        'job_skills': len(job_skills),
        'relevant': len(relevant_candidates),

    }
    return render(request, 'recruiters/job_candidate_search.html', context)


@login_required
def applicant_list(request, slug):
    job = get_object_or_404(Job, slug=slug)
    applicants = Applicants.objects.filter(job=job).order_by('date_posted')
    profiles = []
    for applicant in applicants:
        profile = Profile.objects.filter(user=applicant.applicant).first()
        profiles.append(profile)
    context = {
        'rec_navbar': 1,
        'profiles': profiles,

        'job': job,
    }
    return render(request, 'recruiters/applicant_list.html', context)


@login_required
def selected_list(request, slug):
    job = get_object_or_404(Job, slug=slug)
    selected = Selected.objects.filter(job=job).order_by('date_posted')
    profiles = []
    for applicant in selected:
        profile = Profile.objects.filter(user=applicant.applicant).first()
        profiles.append(profile)
    context = {
        'rec_navbar': 1,
        'profiles': profiles,
        'job': job,
    }
    return render(request, 'recruiters/selected_list.html', context)


@login_required
def select_applicant(request, can_id, job_id):
    job = get_object_or_404(Job, slug=job_id)
    profile = get_object_or_404(Profile, slug=can_id)
    user = profile.user
    selected, created = Selected.objects.get_or_create(job=job, applicant=user)
    applicant = Applicants.objects.filter(job=job, applicant=user).first()
    applicant.delete()
    return HttpResponseRedirect('/hiring/job/{}/applicants'.format(job.slug))


@login_required
def remove_applicant(request, can_id, job_id):
    job = get_object_or_404(Job, slug=job_id)
    profile = get_object_or_404(Profile, slug=can_id)
    user = profile.user
    applicant = Applicants.objects.filter(job=job, applicant=user).first()
    applicant.delete()
    return HttpResponseRedirect('/hiring/job/{}/applicants'.format(job.slug))





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


def create_profile(file):
    text = pdfextract(file) 
    text = str(text)
    text = text.replace("\\n", "")
    text = text.lower()
    #below is the csv where we have all the keywords, you can customize your own
    keyword_dict = pd.read_csv('C:/Users/hp/Desktop/Fenice-Network2/Fenice-Network/recruiters/template_new.csv',sep=";")
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





def calcul_score(request):
    final_database=pd.DataFrame()
    i = 0 
    current_user=request.user.id
    data=Profile.objects.get(user_id=current_user)
    print("data",data)
    print("resume",data.resume)
    print(type(data.resume))
    print(str(data.resume))
    x=str(data.resume)
    f=x.replace("/","\\")
    onlyfiles=[]
    resume="C:/Users/hp/Desktop/Fenice-Network2/Fenice-Network/media/"+str(f)#.split("/")[1]
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

    feature = pd.read_csv("C:/Users/hp/Desktop/Fenice-Network2/Fenice-Network/recruiters/template_new.csv", sep=";")
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

        
    context={"liste":liste}
    return render(request,"recruiters/score.html",context)

    # for i,j in liste_f:
    #     print( str(i)+"'s resume matches about "+ str((float(int(j)/len(liste3))*100))+ "% of the job description.")


   