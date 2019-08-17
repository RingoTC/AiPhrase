<template>
<div class="navbar">
  <a-row class="navRow">
    <a-col :span="6">
      <div class="logo">
        <img src="../assets/logo.png"/>
      </div>
    </a-col>
    <a-col class="navSwitch" :span="12">
      <span><a href="/#/Practice/">AI 智能评估</a></span>
      <span><a class="navSwitchActive" href="/#/Paraphrase/">AI 智能改写</a></span>
    </a-col>
    <a-col class="nav" :span="6">
    <a class="navLink">
        <a-popover>
        <template slot="content">
              <a-button @click="$router.push({path:'/Review'})" block>学习记录</a-button>
              <a-button @click="logout" block>退出</a-button>
        </template>
        <a-avatar class="avatar" size="medium"  icon="user" />
      </a-popover>
    </a>
    </a-col>
  </a-row>
</div>
</template>
<style scope>
  .paraphraser .nav{
    display: flex!important;
    flex-direction: row;
    justify-content: center;
  }
  .paraphraser .navLink{
    color:#ccc;
    margin-left:auto;
  }
  .paraphraser .navLink .avatar{
    margin-left: 15px;
  }
  .paraphraser .navLink:hover{
    color:#fff;
  }
  .paraphraser .navColCenter{
    display:flex!important;
    flex-direction: column;
    justify-content: center;
  }
  .paraphraser .navbar{
    display: flex;
    flex-direction: column;
    justify-content: center;
    background:#194978;
    padding:40px 10vw 10px 10vw;
  }
  .paraphraser .navbar .navRow{
    display: flex;
  }
  .paraphraser .navbar .nav{
    display: flex;
    flex-direction:row;
    justify-content: center;
    align-items: center;
  }
  .paraphraser .navbar .navSwitch{
    text-align: center;
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
  }
  .paraphraser .navbar .navSwitch span a{
    transition: border-color 0.5s,color 0.1s;
    padding-bottom:5px;
    font-size:1.6em;
    color:#ccc;
    margin:0 20px 0 20px;
  }
  .paraphraser .navbar .navSwitch span a:hover{
    color:#fff;
    border-bottom:5px solid #f7f41c;
  }
  .paraphraser .navbar .navSwitch .navSwitchActive{
    color:#fff;
    border-bottom:5px solid #f7f41c;
  }
</style>
<script>
import { setTimeout } from 'timers'
export default {
  name: 'paraphraseNav',
  data () {
    return {
      judgeVisible: false,
      questionType: localStorage.getItem('questionType') || 'm'
    }
  },
  methods: {

    logout () {
      let that = this
      localStorage.clear()
      that.$message.success('退出成功')
      setTimeout(function () {
        that.$router.push({
          path: '/Login/'
        })
      }, 300)
    }
  }
}
</script>
