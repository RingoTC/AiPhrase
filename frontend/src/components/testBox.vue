<template>
<div class="testBox testBoxPin" id="testBox">
    <a-row class="question">
      <a-col :span="24">
        <fixed-header :threshold="200">
        <div class="questionSentence">
          <a-row class="questionInfo">
            <a-col :span="1">
              <a-button @click="$router.go(-1)" class="queButton" shape="circle" icon="left" />
            </a-col>
            <a-col :span="22">
              <div class="questionContext">
                  <inputBox :sentence="question"></inputBox>
              </div>
            </a-col>
            <a-col :span="1">
              <a-button @click="next" class="queButton" shape="circle" icon="right" />
            </a-col>
          </a-row>
        </div>
        </fixed-header>
      </a-col>
    </a-row>
    <a-row class="answer">
      <input placeholder="Type your answer" class="answerInput" type="text" v-model="answer"/>
    </a-row>
    <a-row class="submitanswer">
      <button :class="{gradeLoading:gradeLoadingActive}" @click="submitanswer" id="submitanswer" type="button">
        <span v-if="gradeLoadingActive"><a-icon type="loading" style="margin-right:10px;" />提交中</span>
        <span v-if="!gradeLoadingActive">提交</span>
      </button>
    </a-row>
</div>
</template>

<script>
import FixedHeader from 'vue-fixed-header'
import inputBox from '@/components/inputBox.vue'

export default {
  props: {
    question: {
      type: String,
      required: true
    },
    questinIndex: {
      type: String
    }
  },
  components: {
    FixedHeader,
    inputBox
  },
  data: function () {
    return {
      answer: null,
      gradeLoadingActive: false
    }
  },
  methods: {
    next: function () {
      if (window.history.go(1) === undefined) { this.refreshQue() }
    },
    refreshQue: function () {
      this.$router.push({
        path: '/Practice/' + localStorage.getItem('questionType') + '/' + Math.ceil(Math.random() * 10)
      })
    },
    submitanswer: function () {
      let that = this
      that.gradeLoadingActive = true
      if (that.answer === null || that.answer === '') {
        that.$message.error('不能提交空答案，请重新作答')
        that.gradeLoadingActive = false
      } else if (that.answer === that.question) {
        that.$message.warning('不能照抄答案，请重新作答')
        that.gradeLoadingActive = false
      } else {
        let payload = {
          queID: that.$parent.questionID,
          user_id: localStorage.getItem('userID'),
          ans: that.answer
        }
        that.$http({
          url: '/backend/evaluation/',
          method: 'POST',
          data: JSON.stringify(payload)
        }).then(function (response) {
          that.$parent.renderList = that.$parent.renderList || []
          that.$parent.renderList.unshift(response.data)
          if (response.data.isExc === 'true') { that.$message.success('你的答案很棒，已经被选入参考答案！') }
          that.$message.success('答案提交成功')
          that.gradeLoadingActive = false
          localStorage.setItem('nowExp', 'true')
        }).catch(function (error) {
          console.log(error)
          that.$message.error('提交失败，请检查网络')
          that.gradeLoadingActive = false
        })
      }
    }
  }
}
</script>

<style scope>
.questionSentence.vue-fixed-header--isFixed {
  position: fixed;
  left: 0;
  top: 0;
  width: 100vw;
  background: #194978;
  z-index:1;
  display: flex;
  flex-direction: row;
  justify-content: center;
  padding:20px;
  opacity: 0.9;
  animation:fadeIn 0.5s;
}
@keyframes fadeIn {
   0% {opacity: 0;}
   100% {opacity: 1;}
}
.questionSentence.vue-fixed-header--isFixed .questionInfo{
  width:80vw;
}
.testBoxPin{
  margin-bottom: 5vh;
}
.testBoxPin .question{
  background:url('../assets/practiceBg.png') no-repeat center;
  position: relative;
  height:50vh;
  padding:38vh 12.5% 2vh 12.5%;
}
.testBoxPin .queButton:hover{
  cursor: pointer;
}
.testBox .question .questionSentence .ant-row{
  display: flex;
}
.testBox .question .questionSentence .ant-col-1 button{
    background: none;
    border: 0;
    color: #fff;
}
.testBox .question .questionSentence .ant-row .ant-col-1{
  display: flex;
  justify-content: center;
  flex-direction: column;
}
.testBox .question .questionSentence h1{
  text-align: center;
  color:#fff;
}
.testBoxPin .answer{
  padding:5vh 12.5% 2vh 12.5%;
}
.testBoxPin .answer .answerInput{
  border:0;
  border-bottom: 1px solid #ccc;
  font-size:1.8em;
  width:100%;
  transition: 0.5s;
  padding-bottom:12px;
}
.testBoxPin .answer .answerInput::placeholder{
  color:#999;
}
.testBoxPin .answer .answerInput:focus{
  outline: 0;
  border-bottom: 1px solid #2d93ea;
}
.testBoxPin .submitanswer{
  width:20vw;
  margin:0 auto;
}
.testBoxPin .submitanswer button{
  width: 100%;
  border: 0;
  background: #2d93ea;
  color: #fff;
  padding: 2vh;
  font-size: 1.3em;
  margin-top: 3vh;
  -webkit-transition: 0.3s;
  transition: 0.3s;
}
.testBoxPin .submitanswer button:hover{
  background:#36a0fa;
}
.testBoxPin .submitanswer button:active{
  background:#2876b8;
}
.testBoxPin .questionContext{
  font-size:2em;
  color:#fff;
  white-space: normal;
  text-align: center;
}
.testBoxPin .questionContext .inputBox{
  justify-content: center;
}
</style>
