// regpack_ex.js - Example using the RegPack npm package

const { RegPack } = require('regpack');

// Original JavaScript code to compress (RegPack works on code, not just strings)
const str_wrapper = '"'
const originalCode = `${str_wrapper}HELLO WORLD HELLO WORLD HELLO WORLD HELLO WORLD HELLO WORLD HELLO WORLD HELLO WORLD${str_wrapper}`;
console.log("Original code:", originalCode);
console.log("Original length:", originalCode.length);
var packer = new RegPack();
var options = {
    contextType: 0, contextVariableName: false, crushCopiesFactor: 0, crushGainFactor: 1, crushLengthFactor: 0, crushTiebreakerFactor: 1, hash2DContext: false, hashAllObjects: true, hashAudioContext: false, hashWebGLContext: false, reassignVars: false, timeVariableName: "", useES6: true, varsNotReassigned: "", withMath: false, wrapInSetInterval: false
}
var inputList = packer.runPacker(originalCode, options);
var methodCount = inputList.length;
var bestMethod = 0, bestStage = 0, bestCompression = 1e8;
for (var i = 0; i < methodCount; ++i) {
    var packerData = inputList[i];
    for (var j = 0; j < 4; ++j) {
        var output = (j == 0 ? packerData.contents : packerData.result[j - 1][1]);
        var packedLength = packer.getByteLength(output);
        if (packedLength < bestCompression) {
            bestCompression = packedLength;
            bestMethod = i;
            bestStage = j;
        }
    }
}
var bestOutput = inputList[bestMethod].result[bestStage][1]
// if (typeof bestOutput === "string") {
//     console.log(bestOutput.replaceAll(str_wrapper, ''))
// }

console.log(bestOutput)

/*
_='"}}}~"~HELLO WORLD}~ ~ ';for(i in G='}~')with(_.split(G[i]))_=join(pop());eval(_)
for(_='HELLO WORLDYY Y X"XXXY"';G=/[XY]/.exec(_);)with(_.split(G))_=join(shift());eval(_)
 */