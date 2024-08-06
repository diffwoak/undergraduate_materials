const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  chainWebpack: config => {
    config.module
      .rule('markdown')
      .test(/\.md$/)
      .use('vue-loader')
        .loader('vue-loader')
        .end()
      .use('md-loader')
        .loader(path.resolve(__dirname, 'src/loaders/md-loader.js'))
        .end()
  },
  // module:{
  //   rules: [
  //     {
  //       test: /\.md$/,
  //       use: [
  //         {
  //           loader: 'vue-loader', // 这里的使用的最新的 v15 版本
  //           options: {
  //             compilerOptions: {
  //               preserveWhitespace: false
  //             }
  //           }
  //         },
  //         {
  //           // 这里用到的就是刚写的那个 loader
  //           loader: require.resolve('src/loaders/md-loader.js')
  //         }
  //       ]
  //     }
  //   ]
  // }
})
