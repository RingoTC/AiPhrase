from django.urls import path
from . import views

urlpatterns = [
    # ex:/backend
    path('', views.index, name='backend_view'),
    #    url(r'^predict/', views.predict, name="predict"),
    # ex: /backend/user/1
    path('user/<int:tel_number>', views.get_user_info),
    # ex:/backend/problems/s/1
    path('problems/<str:problem_type>/<int:user_id>', views.get_sentence_for_user),
    # 根据id获取题目
    path('problems/<str:problem_type>/<int:problem_index>/user/<int:user_id>', views.get_sentence_by_id),
    # 提交题目并获取答案
    path('evaluation/', views.evaluate_sentence),
    # 根据题目id获取优秀答案
    path('good_answers/<int:problem_id>', views.get_good_answers),
    path('history_answers/<int:user_id>/page/<int:page_index>', views.get_history_answers_by_page),
    path('statics/user/<int:user_id>', views.get_staistic_for_user),
    path('history_answers/<int:user_id>/problem/<int:problem_id>', views.get_history_answers_by_problem),
    # 获取同义词
    path('syn/<str:word>', views.syn_words),
    # 同义改写
    path('paraphraser/', views.get_paraphrase),
    # 用户等级
    path('userlevel/<int:user_id>', views.get_user_level_by_id)

]
