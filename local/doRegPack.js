#!/usr/bin/env node

// Load dependencies
require('./stringHelper.js');
require('./packerData.js');
const { packer } = require('./regPack.js');

function doRegPack(input) {
    // Get rid of comments and empty lines
    input = input.replace(/([\r\n]|^)\s*\/\/.*|[\r\n]+\s*/g, '');

    // set default options
    var options = {
        withMath: false,
        hash2DContext: false,
        hashWebGLContext: false,
        hashAudioContext: false,
        hashAllObjects: true,
        contextVariableName: false,
        contextType: 0,
        reassignVars: true,
        varsNotReassigned: "",
        crushGainFactor: 1,
        crushLengthFactor: 0,
        crushCopiesFactor: 0,
        crushTiebreakerFactor: 1,
        wrapInSetInterval: false,
        timeVariableName: false,
        useES6: true
    };

    var originalLength = packer.getByteLength(input);
    var inputList = packer.runPacker(input, options);
    var methodCount = inputList.length;

    var bestMethod = 0, bestStage = 0, bestCompression = 1e8;
    for (var i = 0; i < methodCount; ++i) {
        var packerData = inputList[i];
        for (var j = 0; j < 4; ++j) {
            var output = (j == 0 ? packerData.contents : packerData.result[j - 1][1]);
            var packedLength = packer.getByteLength(output);
            if (packedLength < bestCompression && packedLength > 0) {
                bestCompression = packedLength;
                bestMethod = i;
                bestStage = j;
            }
        }
    }
    var bestOutput = inputList[bestMethod];

    if (bestStage > 0) {
        return bestOutput.result[bestStage - 1][1]
    }
    return input
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = doRegPack;
}

// CLI usage
if (require.main === module) {
    const input = process.argv[2] || 'console.log("Hello World!")';
    const result = doRegPack(input);
    console.log(JSON.stringify(result));
}