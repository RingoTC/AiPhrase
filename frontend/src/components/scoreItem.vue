<template>
  <div class="scoreItem">
    <a-row>
    <a-col class="scoreItemName" :span="8">
        <h3>{{rate.name}}</h3>
    </a-col>
    <a-col class="scoreItemValue" :span="16">
        <a-row>
          <a-rate :value="rate.value" disabled />
        </a-row>
    </a-col>
    </a-row>
    <a-row>
      <b class="judgeText">{{rate.description}}</b>
    </a-row>
  </div>
</template>

<script>
export default {
  name: 'scoreItem',
  props: {
    rate: {
      type: Object,
      required: true
    }
  }
}
</script>
<style scope>
.judgeText{
    font-size:12px;
}
.scoreItem{
  height:50px;
  margin:10px 0px 10px 0px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.scoreItem .ant-row{
  display: flex;
  flex-direction: row;
}
.scoreItem .ant-row .scoreItemName{
  font-size:1em;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align:left;
}
.scoreItem .ant-row .scoreItemName h3{
  margin: 0;
}
</style>
