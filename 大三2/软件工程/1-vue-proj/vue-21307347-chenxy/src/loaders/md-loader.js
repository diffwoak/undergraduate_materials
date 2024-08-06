module.exports = function (src) {
    const codeBlocks = src.match(/```javascript([\s\S]*?)```/g)
    const jsSource = codeBlocks.map(codeBlock => {
      return codeBlock.replace(/```javascript/, '').replace(/```/, '')
    }).join('\n')
    const res = (
      `<template>\n` +
      `<h1>`+jsSource+`</h1>\n` +
      `</template>\n`+
      `<script>`+
      jsSource+
      `</script>`
    )
    console.log('loading jsSource...')
    console.log(jsSource)
    
    return res
  }

