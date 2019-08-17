import Vue from 'vue'
import Antd from 'ant-design-vue'
import App from './App.vue'
import router from './router'
import 'ant-design-vue/dist/antd.css'
import axios from 'axios'
import 've-charts/lib/common' // 公共chuck，引入单个图表前需引入公共包
import VeRadarChart from 've-charts/lib/VeRadarChart' // 单个图表chuck
import 've-charts/lib/ve-charts.min.css' // **按需引入** 同样需要引入样式

Vue.prototype.$http = axios
Vue.config.productionTip = false
Vue.use(Antd)

Vue.component('VeRadarChart', VeRadarChart)

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')

router.beforeEach((to, from, next) => {
  if (to.meta.title) {
    document.title = to.meta.title
  }
  let authReq = to.meta.auth
  let loginState = window.localStorage.getItem('loginState')
  if (authReq === true) {
    if (loginState === 'signed') {
      next()
    } else {
      next('/Login')
    }
  } else {
    next()
  }
})
