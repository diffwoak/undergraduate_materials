const path = require('path')

module.exports = {
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
  }
}

