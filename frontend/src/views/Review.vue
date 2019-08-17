<template>
  <div class="Review">
    <div class="dashboard">
      <div class="dashboardMain">
        <a-row>
          <a-col :span="11">
            <div class="dashradar">
              <div id="dashrad">
                <ve-radar-chart :height="260" :data="chartData" :legend="legend" :settings="settings" />
              </div>
            </div>
          </a-col>
          <a-col class="leftOfDash" :span="13">
            <a-row>
              <a-col v-for="(item,index) of rssliced" :key="index" :span="8">
                <a-row>
                  <a-progress gapPosition="top" class="reviewCircle" type="circle" :percent="item[0]" :format="percent => `${percent}`+rsd[index][0]"/>
                </a-row>
                <a-row class="historyInfos">
                  <h4 class="historyInfo"><span class="historyName">{{rsd[index][1]}}</span><span class="historyDet">{{item[1]}}</span></h4>
                  <h4 class="historyInfo"><span class="historyName">{{rsd[index][2]}}</span><span class="historyDet">{{item[2]}}</span></h4>
                  <h4 class="historyInfo"><span class="historyName">{{rsd[index][3]}}</span><span class="historyDet">{{item[3]}}</span></h4>
                </a-row>
              </a-col>
            </a-row>
          </a-col>
        </a-row>
      </div>
      <div class="level">
        <span>Lv.{{level}}</span>
      </div>
    </div>
    <div class="reviewCards">
      <a-row v-for="history in historyRenderList" :key="history.queID">
        <reviewScoreCard :rawData="history"></reviewScoreCard>
      </a-row>
    </div>
    <div class="reviewPage">
    <a-button-group v-if="historyRenderList.length !== 0">
      <a-button @click="prevPage" type="primary">
        <a-icon type="left" />上一页
      </a-button>
      <a-button @click="nextPage" type="primary">
        下一页<a-icon type="right" />
      </a-button>
    </a-button-group>
    </div>
  </div>
</template>

<style scope>
.dashboard{
  width:100%;
  padding-top:6vh;
  background:#1c4b78;
}
.dashboardMain{
  width: 75%;
  margin: 0 auto;
}
.reviewPage{
  width:30vw;
  margin:0 auto;
  margin-top:30px;
  text-align: center;
}
.dashboard .dashboardMain{
  text-align: center;
}
.leftOfDash{
  margin-top:2vh;
}
.dashboard .dashboardMain .reviewCircle span{
  color:#fff;
}
.dashboard .dashboardMain .historyInfos{
  margin:10px 0 10px 0;
}
.dashboard .dashboardMain .historyInfo{
  margin-top:5px;
}
.dashboard .dashboardMain .historyName{
  color:#fff;
}
.dashboard .dashboardMain .historyDet{
  color:#6cd7fb;
  margin-left:15px;
}
.dashboard .dashboardMain .ve-radar{
  margin:0 auto;
}
.dashboard .level{
    position: absolute;
    top: 3%;
    left: 15%;
    font-size: 2em;
    color: #fff;
}
</style>

<script>
// @ is an alias to /src
import reviewScoreCard from '@/components/reviewScoreCard.vue'

export default {
  name: 'Review',
  components: {
    reviewScoreCard
  },
  data: function () {
    return {
      historyRenderList: [],
      chartData: null,
      rs: null,
      rssliced: null,
      rsd: [
        ['分', '≥85', '70~85', '≤70'],
        ['次', '今天', '本周', '本月'],
        ['题', '今天', '本周', '本月']
      ],
      page: parseInt(this.$route.params.pageID),
      totalPage: parseInt(this.$route.params.pageID)
    }
  },
  methods: {
    prevPage () {
      if (this.page <= 1) {
        this.$message.warning('没有前一页了')
      } else {
        this.page = parseInt(this.page) - 1
        this.$router.push({
          path: '/Review/' + this.page
        })
      }
    },
    nextPage () {
      console.log(this.page,this.totalPage)
      if (this.page >= this.totalPage) {
        this.$message.warning('没有后一页了')
      } else {
        this.page = parseInt(this.page) + 1
        this.$router.push({
          path: '/Review/' + this.page
        })
      }
    },
    loadingRadarData () {
      let that = this
      this.$http.get('/backend/statics/user/' + localStorage.getItem('userID')).then(function (response) {
        let rs = response.data.rs
        that.rs = rs
        that.rssliced = rs.slice(1)
        that.chartData = {
          dimensions: [
            { name: '语义', max: 5 },
            { name: '词汇', max: 5 },
            { name: '可读性', max: 5 },
            { name: '句型', max: 5 },
            { name: '语法', max: 5 }
          ],
          measures: [{ name: '用户ID: ' + localStorage.getItem('userID'), data: rs[0] }]
        }
      })
    }
  },
  created () {
    let that = this
    this.legend = { show: false }
    this.tooltip = { show: false }
    this.settings = {
      splitNumber: 5,
      itemStyle: { normal: { areaStyle: {
        color: ['#1d4e7d'],
        shadowColor: '#1d4e7d'
      } } }
    }
    this.$http.get('/backend/userlevel/' + localStorage.getItem('userID')).then(function (response) {
      that.level = response.data.level
    })
    this.loadingRadarData()
  },
  mounted: function () {
    if (!this.$route.params.pageID) {
      this.$router.push({
        path: '/Review/1'
      })
    } else {
      localStorage.setItem('nowQuestion', -1)
      this.page = this.$route.params.pageID
      let that = this
      this.$http.get('/backend/history_answers/' + localStorage.getItem('userID') + '/page/' + this.$route.params.pageID).then(function (response) {
        that.historyRenderList = response.data.rs
        that.totalPage = response.data.page_number
      })
      if (that.historyRenderList === [] || that.historyRenderList === undefined || that.historyRenderList === undefined) {
        that.historyNull = true
      }
    }
  },
  watch: {
    '$route' (to, from) {
      // todo : 重加载渲染列表
      if (!this.$route.params.pageID) {
        this.$router.push({
          path: '/Review/1'
        })
      } else {
        localStorage.setItem('nowQuestion', -1)
        this.page = this.$route.params.pageID
        let that = this
        this.$http.get('/backend/history_answers/' + localStorage.getItem('userID') + '/page/' + this.$route.params.pageID).then(function (response) {
          that.historyRenderList = response.data.rs
        })
        if (that.historyRenderList === [] || that.historyRenderList === undefined || that.historyRenderList === undefined) {
          that.historyNull = true
        }
      }
    }
  }
}
</script>
