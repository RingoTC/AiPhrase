<template>
  <div class="practiceScoreRecp">
    <div class="practiceScoreOfanswer">
      <div class="answerBox">
        <a-row>
          <a-col :span="23">
            <inputBox :sentence="gradeInfomation.customer_answer"/>
          </a-col>
          <a-col v-if="gradeInfomation.detail[4].erros.length !== 0" class="grammarHint" :span="1">
              <a-popover title="语法纠错提示" placement="bottom" overlayStyle="{width:30px}">
              <template slot="content">
                <a-row style="line-height:24px;"  v-for="(error,index) in gradeInfomation.detail[4].erros" :key="index">
                  <p style="font-size:1em;width:240px;">
                    <b style="margin-right:12px;" color="#666">错误</b><span style="color:red">{{error[0]}}</span><br/>
                    <b style="margin-right:12px;" color="#666">参考</b><span v-for="(sug,index) in error[1]" :key="index">[{{sug}}]&#32;</span><br/>
                    <b style="margin-right:12px;" color="#666">原因</b><span>{{error[2]}}</span>
                  </p>
                  <hr color="#ccc"/>
                </a-row>
              </template>
              <a-icon type="exclamation-circle" style="color:#fcf652;font-size:2em" />
            </a-popover>
          </a-col>
        </a-row>
      </div>
    </div>
    <div class="practiceScoreCard">
      <a-row class="rateItem">
        <a-col class="rateScore" :span="6">
          <div class="rate">
            <circle-counter width="160px" height="160px" :dashCount="100" :activeCount="gradeInfomation.rate" :text="gradeInfomation.rate"/>
          </div>
        </a-col>
        <a-col :span="8">
          <a-row v-for="scoreItem in gradeInfomation.detail.slice(0,3)" :key="scoreItem.id">
            <scoreItem :rate="scoreItem"></scoreItem>
          </a-row>
        </a-col>
        <a-col :span="8">
          <a-row v-for="scoreItem in gradeInfomation.detail.slice(-2)" :key="scoreItem.id">
            <scoreItem :rate="scoreItem"></scoreItem>
          </a-row>
        </a-col>
      </a-row>
    </div>
  </div>
</template>

<script>
import scoreItem from '@/components/scoreItem.vue'
import CircleCounter from '@/components/CircleCounter.vue'
import inputBox from '@/components/inputBox.vue'

export default {
  name: 'practiceScoreCard',
  props: {
    gradeInfomation: {
      type: Object,
      required: true
    }
  },
  data: function (params) {
    return {
      selectSmiword: null,
      simRender: ['xxx', 'abc'],
      simLoading: false
    }
  },
  components: {
    scoreItem,
    CircleCounter,
    inputBox
  }
}
</script>
<style scope>
.practiceScoreRecp{
    width:70vw ;
    margin:0 auto;
    animation: fadein 0.5s;
}
@keyframes fadein {
    from { opacity: 0;}
    to   { opacity: 1; }
}
.practiceScoreCard{
  border:1px solid #ccc;
  position: relative;
  margin-bottom:30px;
}
.practiceScoreRecp .practiceScoreOfanswer .answerBox{
  background:#1d4e7d;
  width: 100%;
  color:#fff;
  border:0;
  padding: 12px 10px 12px 10px;
}
.practiceScoreRecp .practiceScoreOfanswer .answerBox .inputBox{
  font-size:1.4em;
}
.practiceScoreCard .rateItem .rateScore{
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height:160px;
  margin:40px 0px 40px 0px;
}
.rate{
  margin:0 auto;
}
.answerBox .ant-row{
  display: flex;
  flex-direction:row;
}
.answerBox .grammarHint{
  display: flex;
  flex-direction:row;
  justify-content: center;
  align-items: center;
}
</style>
