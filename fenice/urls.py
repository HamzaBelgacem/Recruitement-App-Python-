from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from users import views
from django.conf.urls.static import static
from django.conf import settings
from candidates import views as viewsRec
urlpatterns = [
    path('admin/', admin.site.urls),
    path('home/', include('home.urls')),
    path('login/', views.login_form, name='login'),
    path('register/',views.register_form, name="register_form"),

    path('account/', views.account, name='account'),
    path('logout/', auth_views.LogoutView.as_view(template_name='users/logout.html'), name='logout'),
    path('privacy-policy/', views.privacy, name='privacy-policy'),
    path('terms-of-service/', views.terms, name='terms-of-service'),
    path('hiring/pricing/', views.pricing, name='pricing'),
    path('accounts/', include('allauth.urls')),
    path('', include('candidates.urls')),
    path('hiring/', include('recruiters.urls')),
    path('', include('pwa.urls')),

    path('score/',viewsRec.skill_cand,name="skill_cand"),
    path('organize/',viewsRec.organiser,name="organiser"),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

