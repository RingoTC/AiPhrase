<template>
  <div class="Practice">
    <practiceNav/>
    <testBox :questionIndex="$route.params.questionIndex" :question="question"/>
    <div class="practiceMain">
      <a-row v-for="item in renderList" :key="item.record_id">
        <practiceScoreCard :gradeInfomation="item"/>
      </a-row>
    </div>
  </div>
</template>
<script>

// @ is an alias to /src
import practiceScoreCard from '@/components/practiceScoreCard.vue'
import testBox from '@/components/testBox.vue'
import practiceNav from '@/components/practiceNav.vue'

export default {
  name: 'Practice',
  components: {
    practiceScoreCard,
    testBox,
    practiceNav
  },
  data () {
    return {
      question: '加载中...',
      questionID: null,
      renderList: null,
      excAns: null,
      excVisible: false
    }
  },
  mounted: function () {
    let that = this
    let params = that.$route.params
    that.renderList = []
    that.excAns = []
    if (params.questionIndex && params.questionType) {
      that.$http.get('/backend/problems/' + params.questionType + '/' + params.questionIndex + '/user/' + localStorage.getItem('userID')).then(function (response) {
        let data = response.data
        that.question = data.sentence
        that.questionID = data.problem_id
        localStorage.setItem('nowQuestion', data.problem_id)
        if (data.experienced) {
          localStorage.setItem('nowExp', 'true')
          that.$http.get('/backend/history_answers/' + localStorage.getItem('userID') + '/problem/' + that.questionID).then(function (response) {
            that.renderList = response.data.rs
          })
        } else {
          localStorage.setItem('nowExp', 'false')
          that.renderList = null
        }
      })
    } else {
      that.$router.push({
        path: '/Practice/' + localStorage.getItem('questionType') + '/' + Math.ceil(Math.random() * 10)
      })
    }
  },
  watch: {
    '$route' (to, from) {
      let that = this
      let params = that.$route.params
      that.renderList = []
      that.excAns = []
      if (params.questionIndex && params.questionType) {
        that.$http.get('/backend/problems/' + params.questionType + '/' + params.questionIndex + '/user/' + localStorage.getItem('userID')).then(function (response) {
          let data = response.data
          that.question = data.sentence
          that.questionID = data.problem_id
          localStorage.setItem('nowQuestion', data.problem_id)
          if (data.experienced) {
            localStorage.setItem('nowExp', 'true')
            that.$http.get('/backend/history_answers/' + localStorage.getItem('userID') + '/problem/' + that.questionID).then(function (response) {
              that.renderList = response.data.rs
            })
          } else {
            localStorage.setItem('nowExp', 'false')
            that.renderList = null
          }
        })
      } else {
        that.$router.push({
          path: '/Practice/' + localStorage.getItem('questionType') + '/' + Math.ceil(Math.random() * 10)
        })
      }
    }
  }
}
</script>
