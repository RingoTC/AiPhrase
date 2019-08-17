<template>
<div class="inputBox">
  <span v-for="(word,index) in sentence.split(' ')" :key="index">
     <a-popover placement="bottom" title="同义词" trigger="click">
      <template slot="content">
        <center v-if="simLoading"><a-spin /></center>
        <a-alert v-if="queryWordNull" :showIcon="false" message="暂未查询到同义词" banner />
        <div class="syns">
          <span class="synWord" v-for="(sim,index) in simRender" :key="index">{{sim}}</span>
        </div>
      </template>
      <span @click="querySim(sentence,index)" class="similarword">{{word}}</span>
    </a-popover>
  </span>
</div>
</template>
<style scope>
.inputBox{
  display: -webkit-flex;
  display: flex;
  -webkit-flex-wrap: wrap;
  flex-wrap: wrap;
}
.syns{
  width:200px;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-wrap: wrap;
  flex-wrap: wrap;
}
.synWord{
  background-color: #4a6ee0;
  color: #fff;
  font-size:1em;
  padding:3px;
  margin:4px;
  border-radius:3px;
  transition-duration: .2s;
  text-align:center;
}
.synWord:hover{
  background-color: #4974ff;
  cursor: pointer;
}
.similarword{
  margin-right:6px;
  word-wrap: break-word;
  font-family:serif;
}
</style>

<script>
export default {
  name: 'inputBox',
  props: {
    sentence: {
      type: String
    }
  },
  data: function () {
    return {
      simLoading: true,
      simRender: [],
      queryWord: null,
      queryWordNull: false
    }
  },
  methods: {
    querySim: function (sentence,index) {
      let that = this
      that.simRender = null
      that.queryWordNull = false
      let payload = {
        sentence: sentence,
        id: index + 1
      }
      that.$http({
        url: '/backend/syn/',
        method: 'POST',
        data: JSON.stringify(payload)
      }).then(function (response) {
        let syn = response.data.syn
        if (syn.length === 0) {
          // 未查询到同义词
          that.queryWordNull = true
          that.simLoading = false
        } else {
          that.simLoading = false
          that.queryWordNull = false
          that.simRender = response.data.syn
        }
      })
      /*
      that.$http.get('/backend/syn/' + word).then(function (response) {
        let syn = response.data.syn
        if (syn.length === 0) {
          // 未查询到同义词
          that.queryWordNull = true
          that.simLoading = false
        } else {
          that.simLoading = false
          that.queryWordNull = false
          that.simRender = response.data.syn
        }
      })
      */
    }
  }
}
</script>
