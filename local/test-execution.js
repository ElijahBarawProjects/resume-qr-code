const doRegPack = require('./doRegPack.js');

// Test with simple JavaScript code
const testCode = `console.log("Hello World!");`;
console.log('Original:', testCode);

const compressed = doRegPack(testCode);
console.log('Compressed:', compressed);

// Test if the compressed code actually runs
try {
    console.log('\n--- Testing execution ---');
    eval(compressed);
} catch (error) {
    console.error('Execution failed:', error.message);
}