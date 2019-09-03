from django.test import TestCase
from .models import User
from .views import get_user_info
from django.test import Client


# Create your tests here.

class UserApiTests(TestCase):
    def test_query_registered_user(self):
        '''
        test_query_registered_user
        '''
        # 测试get_user_info()
        url = 'http://52.80.106.20:8001/backend/user/18323819603'
        client = Client()
        response = client.get(url)
        print(response)
        assert response == '{"signed": true, "userID": 2, "phoneNumber": "18323819602"}'


