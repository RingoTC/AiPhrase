<template>
  <div v-if="loginState !== 'signed'"  id="login" class="flex-columns Login">
    <div class="loginLogo">
      <img src="../assets/logo.png"/>
    </div>
    <div class="loginMain">
      <h1>验证码登录</h1>
      <span class="inputCompact">
        <input v-model="phoneArea" class="phoneArea" type="text" style="width:30%" placeholder="国家区号" value="+86"/>
        <input @click="removeErrorLint('phone')" v-model="phoneNumber" class="phoneNumber" type="text" style="width:70%" placeholder="请输入手机号码"/>
        <a id="phoneLint" v-if="phoneError" class="signRightHint codeError" href="#/Login">请输入正确的手机号码</a>
      </span>
      <span class="inputCompact">
        <input @click="removeErrorLint('Verifi')" v-model="VerifiCode" class="VerifiCode" type="text" style="width:100%" placeholder="请输入手机验证码"/>
        <a id="getVerifiCode" v-if="!VerifiError" v-on:click="getVerifiCode" class="signRightHint">{{(countDown)?countDown+"s 后获取":"获取验证码"}}</a>
        <a id="VerifiCodeLint" v-if="VerifiError" v-on:click="getVerifiCode" class="signRightHint codeError">请输入正确的验证码</a>
      </span>
      <button :class="{loginLoadding:loginLoaddingActive}" @click="login" id="loginButton" type="button">
        <span v-if="loginLoaddingActive"><a-icon type="loading" style="margin-right:10px;" />登陆中</span>
        <span v-if="!loginLoaddingActive">登陆</span>
      </button>
    </div>
  </div>
</template>
<style scope>
  .Login{
      width:100%;
      height: 100vh;
      background-image: linear-gradient(to bottom, #1d4e7d 0%, #1d4e7d 50%, #ffffff 50%, #ffffff 100%);
      padding:40px 12.5% 10px 12.5%;
  }
  .Login img{
    width:15vw;
  }
  .Login .loginMain{
    width:30vw;
    margin:0 auto;
    margin-top:14%;
    background:#fff;
  }
  .Login .loginMain{
    text-align:center;
    box-shadow: 0px 6px 28.2px 1.8px rgba(0, 0, 0, 0.08);
    border-radius: 5px;
    padding:20px;
  }
  .inputCompact{
    display:flex;
    flex-direction: row;
    position: relative
  }
  .inputCompact input[type=text]{
    border:0;
    border-bottom: 1px solid #e6e6e6;
    color:#333;
    height:50px;
    margin-bottom: 10px;
    background: none;
    font-size:1.3em;
    transition: 0.5s;
  }
  .inputCompact input[type=text]::placeholder{
    color:#999;
  }
  .inputCompact input[type=text]:focus{
    border-bottom: 1px solid #2d93ea;
  }
  .inputCompact .phoneArea{
    text-align: center;
    border-right: none!important;
  }
  .inputCompact .phoneNumber{
    border-left:none!important;
  }
  .inputCompact .phoneNumber,.inputCompact .VerifiCode{
    padding-left:20px;
  }
  .inputCompact input[type=text]::placeholder{
    color:#ccc;
  }
  .inputCompact input[type=text]:focus{
    outline: none;
  }
  .inputCompact .signRightHint{
    font-size:1em;
    position: absolute;
    top: 25px;
    right: 10px;
    padding: 4px 0;
    color: #175199;
    -webkit-transform: translateY(-50%);
    transform: translateY(-50%);
    color:#ccc;
    text-decoration:none;
    transition: cubic-bezier(0.755, 0.05, 0.855, 0.06);
    background:#fff;
  }
  .inputCompact .signRightHint::before{
    position: absolute;
    top: 0;
    left: -36px;
    width: 36px;
    height: 26px;
    background: -webkit-gradient(linear,left top,right top,from(hsla(0,0%,100%,0)),to(#fff));
    background: linear-gradient(90deg,hsla(0,0%,100%,0),#fff);
    content: "";
  }
  .inputCompact .signRightHint:hover{
    color:#175199;
  }
  #login #loginButton{
    width:100%;
    border:0;
    background: #2d93ea;
    color:#fff;
    padding:1.6vh;
    font-size:1em;
    margin-top:3vh;
    transition: 0.3s;
  }
  #login #loginButton:hover{
    background:#36a0fa;
  }
  #login #loginButton:active{
    background: #2876b8;
  }
  #login .codeError{
    color:red;
  }
  #login #getVerifiCode{
    color:#2d93ea;
    font-size:1.2em;
  }
  #login .loginLoadding{
    cursor: not-allowed;
    background:#36a0fa;
  }
  #login .loginLoadding:active{
    background:#36a0fa!important;
  }
</style>
<script>
import { setInterval, clearInterval, setTimeout } from 'timers'
export default {
  name: 'login',
  mounted () {
    if (localStorage.getItem('signed') === 'signed') {
      this.$router.push({ path: '/' })
    }
  },
  data () {
    return {
      phoneNumber: localStorage.getItem('phoneNumber') || null,
      phoneArea: '+86',
      VerifiCode: null,
      loginState: localStorage.getItem('loginState') || 'notSigned',
      countDown: null,
      VerifiError: false,
      phoneError: false,
      loginLoaddingActive: false
    }
  },
  methods: {
    login: function (events) {
      let that = this
      let phoneCheck = /^[1](([3][0-9])|([4][5-9])|([5][0-3,5-9])|([6][5,6])|([7][0-8])|([8][0-9])|([9][1,8,9]))[0-9]{8}$/
        .test(this.phoneNumber)
      let areaCheck = (this.phoneArea.length > 2)
      let VerifiCodeCheck = (this.VerifiCode === '3773')
      if (!phoneCheck || !areaCheck) {
        this.phoneError = true
      } else if (!VerifiCodeCheck) {
        this.VerifiError = true
      } else {
        that.loginLoaddingActive = true
        that.$http.get('/backend/user/' + that.phoneNumber).then(function (response) {
          let data = response.data
          if (data.signed === true) {
            localStorage.setItem('loginState', 'signed')
            localStorage.setItem('phoneNumber', data.phoneNumber)
            localStorage.setItem('userID', data.userID)
            localStorage.setItem('questionType', 'm') // 默认显示中句
            that.$message.success('登录成功，请开始你的表演', 10)
            that.loginLoaddingActive = false
            that.$router.push({
              path: '/Practice/'
            })
          }
        }).catch(function (error) {
          that.$$message.error('登录失败', 10)
        })
      }
    },
    getVerifiCode: function () {
      let that = this
      if (!that.countDown) {
        this.countDown = 60
        var counter = setInterval(function () {
          that.countDown = that.countDown - 1
          if (that.countDown <= 0) {
            clearInterval(counter)
            that.countDown = null
          }
        }, 1000)
      }
    },
    removeErrorLint: function (opt) {
      if (opt === 'Verifi') { this.VerifiError = false }
      if (opt === 'phone') { this.phoneError = false }
    }
  }
}
</script>
