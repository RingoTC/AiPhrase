import Vue from 'vue'
import Router from 'vue-router'
import Practice from './views/Practice.vue'
import Review from './views/Review.vue'
import Login from './views/Login.vue'
import Paraphrase from './views/Paraphrase.vue'
import vueg from 'vueg'

Vue.use(Router)
Vue.use(vueg, new Router())

export default new Router({
  mode: 'hash',
  routes: [
    {
      path: '/',
      redirect: '/Practice/'
    },
    {
      path: '/Paraphrase',
      name: 'Paraphrase',
      component: Paraphrase,
      meta: {
        auth: true,
        title: 'Paraphrase | AiPhrase'
      }
    },
    {
      path: '/Practice',
      name: 'Practice',
      component: Practice,
      meta: {
        auth: true,
        title: 'Practice | AiPhrase'
      }
    },
    {
      path: '/Practice/:questionType/:questionIndex/',
      name: 'PracticeWithIndex',
      component: Practice,
      meta: {
        auth: true,
        title: 'Practice | AiPhrase'
      }
    },
    {
      path: '/Review/',
      name: 'ReviewWith',
      component: Review,
      meta: {
        auth: true,
        title: 'Review | AiPhrase'
      }
    },
    {
      path: '/Review/:pageID/',
      name: 'ReviewWithPageID',
      component: Review,
      meta: {
        auth: true,
        title: 'Review | AiPhrase'
      }
    },
    {
      path: '/Login',
      name: 'Login',
      component: Login,
      meta: {
        transition: 'fade-in-up',
        title: 'Login | AiPhrase'
      }
    }
  ]
})
