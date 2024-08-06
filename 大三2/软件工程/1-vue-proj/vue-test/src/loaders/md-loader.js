module.exports = function (src) {
    const codeBlocks = src.match(/```javascript([\s\S]*?)```/g)
    src = codeBlocks.map(codeBlock => {
      return codeBlock.replace(/```javascript/, '').replace(/```/, '')
    }).join('\n')
    console.log(src)
    const res = (
      `<template>\n` +
      `<h1>`+src+`</h1>\n` +
      `</template>`+
      ``+
      `<script>`+
      `src`+
      `</script>`
    )
    return res
  }