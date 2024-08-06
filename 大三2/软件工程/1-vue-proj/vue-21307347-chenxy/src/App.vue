<template>
  <HelloWorld msg="Hello 21307347 陈欣宇"/>
  <h1>{{ message }}</h1>

</template>

<script>
import HelloWorld from './components/HelloWorld.vue'
import MyMarkdownLoaderExample from './my-markdown-loader-example.js'

export default {
  components: {
    HelloWorld,
  },
  data() {
    return {
      message: ''
    };
  },
  methods: {
    fetchMessage() {
      // 实例化 MyMarkdownLoaderExample 并调用 getMsg 方法
      const loader = new MyMarkdownLoaderExample();
      this.message = loader.getMsg();
    }
  },
  mounted() {
    // 组件挂载后调用 fetchMessage 方法
    this.fetchMessage();
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
