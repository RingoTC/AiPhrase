<template>
<div class="navbar">
  <a-row class="navRow">
    <a-col :span="6">
      <div class="logo">
        <img src="../assets/logo.png"/>
      </div>
    </a-col>
    <a-col class="navSwitch" :span="12">
      <span><a href="/#/Practice/" class="navSwitchActive">AI 智能评估</a></span>
      <span><a href="/#/Paraphrase/">AI 智能改写</a></span>
    </a-col>
    <a-col class="nav" :span="6">
    <a-select class="questionTypeSele navColCenter" :defaultValue="questionType" @change="handleChange">
      <a-select-option value="l">长句</a-select-option>
      <a-select-option value="m">中句</a-select-option>
      <a-select-option value="s">短句</a-select-option>
    </a-select>
    <a class="navLink navColCenter" @click="showModal">评分标准</a>
    <a-modal
        title="评分标准"
        v-model="judgeVisible"
        @ok="handleOk"
      >
      <center>满分（100分）</center>
      <h4>分档标准</h4>

      <p><b>90-100分：</b>准确理解原句含义，改写语句与原句语义相似，句型和词汇选用恰当、丰富，语句自然流畅，语法、拼写等错误极少。</p>

      <p><b>75-90分：</b>理解原句含义，改写语句与原句语义基本相似，语句比较自然流畅，有少量语法、拼写等错误。</p>

      <p><b>60-75分：</b>部分理解原句含义，改写语句丢失大部分原句信息，句型和词汇单一，有少量语法、拼写等错误。</p>

      <p><b>40-60分：</b>不理解原句含义，改写语句与原句语义不相似，句型和词汇单一，语法、拼写等错误较多。</p>

      <p><b>0-40分：</b>不理解原句含义，改写语句与原句语义无关，句型和词汇单一，有大量语法、拼写等错误。</p>
    </a-modal>
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
  .Practice .navbar{
      margin: auto;
      position: absolute;
      top: 40px; left: 0; right: 0;
      z-index: 3;
      width:80vw;
  }
  .Practice .navbar .nav{
      text-align:right;
  }
  .Practice .nav{
    display: flex!important;
    flex-direction: row;
    justify-content: center;
  }
  .Practice .navLink{
    color:#ccc;
    margin-left:auto;
  }
  .Practice .navLink .avatar{
    margin-left: 15px;
  }
  .Practice .navLink:hover{
    color:#fff;
  }
  .Practice .navColCenter{
    display:flex!important;
    flex-direction: column;
    justify-content: center;
  }
  .Practice .questionTypeSele [role=combobox]{
    background:none;
    border:none;
    border-radius: 0;
    color:#eee;
  }
  .Practice .questionTypeSele [role=combobox]:active{
    border:0!important;
  }
  .Practice .questionTypeSele span{
    color:#fff!important;
  }
  .Practice .navbar{
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .Practice .navbar .navRow{
    display: flex;
  }
  .Practice .navbar .nav{
    display: flex;
    flex-direction:row;
    justify-content: center;
    align-items: center;
  }
  .Practice .navbar .navSwitch{
    text-align: center;
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
  }
  .Practice .navbar .navSwitch span a{
    margin:0 20px 0 20px;
    transition: border-color 0.5s,color 0.1s;
    padding-bottom:5px;
    font-size:1.6em;
    color:#ccc;
  }
  .Practice .navbar .navSwitch span a:hover{
    color:#fff;
    border-bottom:5px solid #f7f41c;
  }
  .Practice .navbar .navSwitch .navSwitchActive{
    color:#fff;
    border-bottom:5px solid #f7f41c;
  }
</style>
<script>
import { setTimeout } from 'timers'
export default {
  name: 'practiceNav',
  data () {
    return {
      judgeVisible: false,
      questionType: localStorage.getItem('questionType') || 'm'
    }
  },
  methods: {
    showModal () {
      this.judgeVisible = true
    },
    handleOk (e) {
      this.judgeVisible = false
    },
    handleChange (value) {
      localStorage.setItem('questionType', value)
      this.$router.push({
        path: '/Practice/'
      })
    },
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
