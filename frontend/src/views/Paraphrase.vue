<template>
    <div class="paraphraser">
        <paraphraseNav></paraphraseNav>
        <div class="paraphraserMain">
            <textarea v-model="paraSentence" placeholder="Type in anything that you want here. Then press the Paraphrase It button to paraphrase it." class="inputSentence"/>
            <button :class="{paraphrasing:paraphrasingActive}" @click="getParaphrase" id="submitParaphrase" type="button">
                <span v-if="paraphrasingActive"><a-icon type="loading" style="margin-right:10px;" />Paraphrasing</span>
                <span v-if="!paraphrasingActive">Paraphrase it</span>
            </button>
            <div class="listOfPara">
              <div v-for="(para,index) in paraRenderList" :key="index" class="paraList">
                <input :id="'paraText'+index" :value="para"/>
                <a-button shape="circle" @click="copyText(index)" icon="copy"></a-button>
              </div>
            </div>
        </div>
    </div>
</template>
<style scoped>
.paraphraser .paraphraserMain{
    width:60vw;
    margin:0 auto;
    padding: 30px 0 0 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.paraphraser #submitParaphrase{
  width: 20vw;
  border: 0;
  background: #2d93ea;
  color: #fff;
  padding: 2vh;
  font-size: 1.3em;
  -webkit-transition: 0.3s;
  transition: 0.3s;
  margin:0 auto;
  margin-bottom:30px;
}
.paraphraser #submitParaphrase:hover{
  background:#36a0fa;
}
.paraphraser #submitParaphrase:active{
  background:#2876b8;
}
.paraphraser .paraphraserMain .inputSentence{
  width:100%;
  height:200px;
  line-height:40px;
  font-size:1.5em;
  padding-left:12px;
  resize:none;
  border:1px solid #ccc;
  outline:none;
  margin-bottom:30px;
}
.paraphraser .paraphraserMain input{
  width:100%;
  height:40px;
  line-height:40px;
  font-size:1.2em;
  padding-left:12px;
  border:0;
  outline:0;
  border-bottom: 1px solid #ccc;
  margin-bottom:10px;
  transition: 0.3s;
  padding-right:40px;
}
.paraphraser .paraphraserMain input:focus{
  border-bottom: 1px solid #2d93ea;
}
.paraphraser .paraphraserMain .listOfPara{
  position: relative;
}
.paraphraser .paraphraserMain .listOfPara button{
  transition: 0.3s;
  opacity: 0.5;
  position: absolute;
  right: 0;
  margin-top: 20px;
  color: #175199;
  -webkit-transform: translateY(-50%);
  transform: translateY(-50%);
  background:#fff;
}
.paraphraser .paraphraserMain .listOfPara button:hover{
  opacity: 1;
}
</style>

<script>
import paraphraseNav from '@/components/paraphraseNav.vue'
export default {
  name: 'Paraphrase',
  components: {
    paraphraseNav
  },
  data () {
    return {
      paraphrasingActive: false,
      paraRenderList: []
    }
  },
  methods: {
    copyText: function (index) {
      let ele = document.querySelector('#paraText' + index)
      ele.select()
      document.execCommand('copy')
      this.$message.success('已复制')
    },
    getParaphrase: function (sentence) {
      let that = this
      that.paraphrasingActive = true
      if (that.paraSentence === '' || that.paraSentence === undefined) {
        that.$message.warning('请输入需要改写的语句')
        that.paraphrasingActive = false
      } else {
        let payload = {
          sentence: that.paraSentence
        }
        that.$http({
          url: '/backend/paraphraser/',
          method: 'POST',
          data: JSON.stringify(payload)
        }).then(function (response) {
          that.paraRenderList = response.data.paraphrase
          that.paraphrasingActive = false
        }).catch(function (error) {
          that.$message.warning('提交失败，请检查网络')
          that.paraphrasingActive = false
        })
      }
    }
  }
}
</script>
