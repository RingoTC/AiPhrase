<template>
  <div id="app">
    <div class="onlyPC">
      <h1>请使用PC访问本站</h1>
      <h3>Ai Phrase</h3>
    </div>
    <div class="referans">
      <a-drawer
        title="参考答案"
        placement="bottom"
        :closable="true"
        @close="referClose"
        :visible="referVis"
        :mask="true"
        maskStyle="opacity:0"
        getContainer=".container"
      >
        <li v-for="(ans,index) in goodans" :key="index">
          <span class="goodans">{{ans[1]}}</span>
          <span class="goodansscore">
            {{ans[0]}}分
          </span>
        </li>
      </a-drawer>
    </div>
    <div class="container">
      <router-view v-transition></router-view>
    </div>
    <div v-if="showSwitcher" class="switcher">
      <a-row>
        <button class="switch" type="button" @click="switcher">
          <a-icon type="swap" />
        </button>
      </a-row>
      <a-row>
        <button v-if="showReferanw" class="referanw" type="button" @click="referanswer">
          <svg
          xmlns="http://www.w3.org/2000/svg"
          xmlns:xlink="http://www.w3.org/1999/xlink"
          width="28px" height="27px">
          <path fill-rule="evenodd"  fill="rgb(255, 255, 255)"
          d="M27.522,16.761 L19.514,26.280 C19.165,26.697 18.662,26.956 18.126,26.998 C18.070,27.002 18.020,27.004 17.965,27.004 C17.480,27.004 17.013,26.827 16.640,26.506 L11.753,22.249 C10.901,21.508 10.802,20.207 11.534,19.342 C12.265,18.477 13.548,18.379 14.402,19.120 L17.743,22.029 L24.424,14.085 C25.153,13.217 26.438,13.112 27.291,13.853 C28.148,14.594 28.248,15.895 27.522,16.761 ZM9.231,19.309 L5.329,19.309 C4.687,19.309 4.168,18.809 4.168,18.158 C4.168,17.509 4.687,17.007 5.330,17.006 L11.235,17.006 C10.773,17.249 10.348,17.560 9.988,17.984 C9.645,18.392 9.394,18.885 9.231,19.309 ZM15.787,7.745 L5.330,7.745 C4.688,7.745 4.169,7.215 4.169,6.565 C4.169,5.914 4.687,5.384 5.330,5.384 L15.788,5.384 C16.429,5.384 16.948,5.914 16.946,6.565 C16.946,7.215 16.430,7.745 15.787,7.745 ZM15.787,11.680 L5.330,11.680 C4.688,11.680 4.169,11.147 4.169,10.499 C4.169,9.847 4.687,9.318 5.330,9.318 L15.788,9.318 C16.429,9.318 16.948,9.848 16.946,10.499 C16.946,11.147 16.430,11.680 15.787,11.680 ZM16.950,18.170 C16.950,18.310 16.927,18.441 16.883,18.562 L15.724,17.556 C15.474,17.335 15.202,17.129 14.915,17.007 L15.788,17.007 C16.429,17.007 16.948,17.522 16.950,18.170 ZM4.169,14.405 C4.169,13.754 4.687,13.254 5.330,13.254 L15.788,13.254 C16.429,13.254 16.948,13.754 16.946,14.405 C16.946,15.054 16.430,15.555 15.787,15.555 L5.330,15.555 C4.688,15.555 4.169,15.055 4.169,14.405 ZM18.565,4.154 C18.565,3.144 17.753,2.357 16.758,2.357 L4.148,2.357 C3.153,2.357 2.329,3.144 2.329,4.154 L2.329,20.533 C2.329,21.543 3.153,22.335 4.148,22.335 L9.358,22.335 C9.596,22.880 9.952,23.391 10.425,23.805 L11.506,24.757 L2.358,24.757 C1.047,24.757 0.000,23.678 0.000,22.351 L0.000,2.302 C0.000,1.028 1.004,-0.005 2.261,-0.005 L18.552,-0.005 C19.860,-0.005 20.894,1.072 20.894,2.400 L20.894,15.071 L18.565,17.878 L18.565,4.154 Z"/>
          </svg>
        </button>
      </a-row>
      <a-row>
          <a-back-top />
      </a-row>
    </div>
    <div v-if="loginState === 'signed'" class="footer">

    </div>
  </div>
</template>
<style scope>
@import './assets/style.css';
.footer{
  width:100%;
  height:20vh;
  background:url("./assets/logo-dark.png") no-repeat center;
  position: relative;
  bottom: 0;
  z-index:-1;
}
.switcher{
  position: fixed;
  right:3%;
  top: 70%;
  right:calc(100% - 97vw);
}
.switcher button{
  border:0;
  background:0;
  height: 40px;
  width: 40px;
  border-radius: 50%;
  color: #fff;
  text-align: center;
  font-size: 20px;
  background:#333;
  margin-bottom: 10px;
  transition: 0.5s;
  outline: none;
  opacity: 0.5;
  display: flex;
  justify-content: center;
  align-items: center;
}
.switcher button:hover{
  background:#000;
  opacity: 0.8;
  cursor: pointer;
}
.switcher .ant-back-top{
  position: absolute;
  left: 0;
  top: 0;
  right: 0;
  bottom: 0;
}
.goodans{
  font-size:1.3em;
  width:80%;
}
.goodansscore{
  width:20%;
  font-size:1.3em;
  text-align: right;
  float: right;
}
.ant-drawer-wrapper-body{
  width: 75%;
  margin: 0 auto;
}
.ant-drawer-close{
  right:12.5%;
}
</style>
<script>
export default {
  name: 'app',
  created () {
    document.title = 'Ai Phrase'
    if (window.localStorage.getItem('loginState') !== 'signed') {
      this.$router.push({
        path: '/Login'
      })
    }
  },
  mounted: function () {
    this.loginState = localStorage.getItem('loginState') || 'notSigned'
    let nowPath = this.$route.path
    if (nowPath.indexOf('/Practice') === 0) {
      this.showSwitcher = true
      this.showReferanw = true
    } else if (nowPath.indexOf('/Review') === 0) {
      this.showSwitcher = true
      this.showReferanw = false
    } else {
      this.showSwitcher = false
      this.showReferanw = false
    }
  },
  data () {
    return {
      showSwitcher: true,
      showReferanw: true,
      referVis: false,
      goodans: null,
      loginState: localStorage.getItem('loginState') || 'notSigned'
    }
  },
  watch: {
    '$route' (to, from) {
      this.loginState = localStorage.getItem('loginState') || 'notSigned'
      let nowPath = this.$route.path
      if (nowPath.indexOf('/Practice') === 0) {
        this.showSwitcher = true
        this.showReferanw = true
      } else if (nowPath.indexOf('/Review') === 0) {
        this.showSwitcher = true
        this.showReferanw = false
      } else {
        this.showSwitcher = false
        this.showReferanw = false
      }
    }
  },
  methods: {
    switcher: function () {
      let nowPath = this.$route.path
      if (nowPath.indexOf('/Practice') === 0) { // 当前在练习页面
        this.$router.push({
          path: '/Review/1'
        })
      } else { // 当前在复习/登录页面
        this.$router.push({
          path: '/Practice/' + localStorage.getItem('questionType') + '/' + Math.ceil(Math.random() * 10)
        })
      }
    },
    referanswer: function () {
      let that = this
      if (localStorage.getItem('nowExp') === 'true') {
        that.$message.success('已获取优秀答案')
        that.$http.get('/backend/good_answers/' + localStorage.getItem('nowQuestion')).then(function (response) {
          that.goodans = response.data.excAns_scores
          that.referVis = true
        }).catch(function (error) {
          that.$message.error('获取优秀答案失败，请检查网络')
        })
      } else {
        that.$message.error('尚未提交答案，不能查询优秀答案')
      }
    },
    referClose () {
      this.referVis = false
    }
  }
}
</script>
