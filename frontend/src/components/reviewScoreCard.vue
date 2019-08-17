<template>
  <div class="reviewScoreCard">
    <div class="reviewScoreCardHeader">
      <inputBox :sentence="rawData.problem"/>
    </div>
    <div class="reviewScoreCardAns">
      <h4><span>我的答案</span><span class="redoQue"><a :href="'/#/Practice/'+rawData.problem_type+'/'+rawData.problem_index"><a-icon style="color:#333" type="edit" /></a></span></h4>
      <a-row v-for="(ans,index) in rawData.history" :key="index">
        <a-row class="flexRowCenter">
          <a-col :span="22">
            <inputBox :sentence="ans[0]"/>
          </a-col>
          <a-col class="gradeTag" :span="2">
            <span>{{ans[1]}}</span>
          </a-col>
        </a-row>
      </a-row>
    </div>
  </div>
</template>

<style scope>
.reviewScoreCard{
    width:70vw ;
    margin:0 auto;
    animation: fadein 0.5s;
    border:1px solid #ccc;
    margin-top: 30px;
}
.reviewScoreCard .reviewScoreCardHeader{
  background:#1d4e7d;
  width: 100%;
  color:#fff;
  border:0;
  padding: 12px 10px 12px 10px;
}
.reviewScoreCard .reviewScoreCardAns{
  padding: 10px 20px 10px 20px;
}
.reviewScoreCard .reviewScoreCardAns .similarword{
  font-size: 1.2em;
}
.reviewScoreCard .reviewScoreCardAns .similarword:hover{
  background: #6cd7fb;
}
@keyframes fadein {
    from { opacity: 0;}
    to   { opacity: 1; }
}
.gradeTag{
  text-align: right;
  font-size:1.5em;
  color:#1d4e7d;
}
.flexRowCenter{
  display: flex;
  justify-content: center;
  flex-direction:row;
}
.reviewScoreCard .redoQue{
  float:right;
  font-size:1.3em;
  cursor:pointer;
}
</style>

<script>
import inputBox from '@/components/inputBox.vue'
export default {
  name: 'reviewScoreCard',
  components: {
    inputBox
  },
  props: {
    rawData: {
      type: Object,
      required: true
    }
  }
}
</script>
