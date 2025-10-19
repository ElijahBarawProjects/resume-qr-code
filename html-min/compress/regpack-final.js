#!/usr/bin/env node

/**
 * Second-stage compression: RegPack + QR code generation
 * Takes output from Rust compressor (num_replacements, last_symbol, wordlist, body)
 * and applies RegPack compression
 *
 * Usage: node regpack-final.js <intermediates-file> <output-html> <output-qr>
 */

const fs = require('fs');
const path = require('path');

// Parse command line arguments
const args = process.argv.slice(2);
if (args.length < 3) {
    console.error('Usage: node regpack-final.js <intermediates-file> <output-html> <output-qr>');
    console.error('');
    console.error('Arguments:');
    console.error('  <intermediates-file>  File with format: num_replacements\\nlast_symbol\\nwordlist\\n`body`');
    console.error('  <output-html>         Path for final compressed HTML output');
    console.error('  <output-qr>           Path for QR code PNG output');
    process.exit(1);
}

const [intermediatesFile, outputHtml, outputQr] = args;

// Load RegPack
const doRegPack = require(path.join(__dirname, '../../local/doRegPack.js'));

// Read intermediates file
const intermediatesContent = fs.readFileSync(intermediatesFile, 'utf8');
const lines = intermediatesContent.split('\n');

if (lines.length < 4) {
    console.error('Error: Intermediates file must have 4 lines:');
    console.error('  Line 1: num_replacements');
    console.error('  Line 2: last_symbol');
    console.error('  Line 3: replacement_list (wordlist with backticks)');
    console.error('  Line 4: `compressed` (body with backticks)');
    process.exit(1);
}

// Parse the intermediates
const num_replacements = parseInt(lines[0].trim(), 10);
const last_symbol = parseInt(lines[1].trim(), 10);
const replacement_list = lines[2].trim(); // Already has backticks from Rust
const compressed_with_backticks = lines[3].trim();

// Remove backticks from compressed body for RegPack processing
const compressed = compressed_with_backticks.replace(/^`|`$/g, '');

console.log('Stage 1 (Rust) results:');
console.log('  Number of replacements:', num_replacements);
console.log('  Last symbol code:', last_symbol);
console.log('  Wordlist entries:', replacement_list.split('|').length);
console.log('  Body size (pre-RegPack):', compressed.length, 'bytes');

// Run RegPack on the body
let packed = doRegPack(compressed);

// Clean up RegPack output
if (packed.endsWith('eval(_)')) {
    packed = packed.slice(0, -7);
}
packed = packed.replaceAll("\\`", "");

console.log('\nStage 2 (RegPack) results:');
console.log('  Body size (post-RegPack):', packed.length, 'bytes');
console.log('  RegPack savings:', compressed.length - packed.length, 'bytes');

// Build final HTML using the same template as format_to_string() in compressor.rs
// Template: <meta charset="utf-8"><script>onload=()=>{let w={replacement_list}.split("|"),h=`{compressed}`;for(i=0;i<{num_replacements};)h=h.replaceAll("~"+String.fromCharCode({last_symbol}-i),w[i++]);document.body.innerHTML=h};</script>
//
// But we replace the compressed body with the RegPack'd version
const finalHtml = `<meta charset="utf-8"><script>onload=()=>{let w=${replacement_list}.split("|");${packed}for(i=0;i<${num_replacements};)_=_.replaceAll("~"+String.fromCharCode(${last_symbol}-i),w[i++]);document.body.innerHTML=_};</script>`;

// Write the final HTML
fs.writeFileSync(outputHtml, finalHtml);
console.log('\nFinal HTML written to:', outputHtml);
console.log('Final HTML size:', finalHtml.length, 'bytes');

// Calculate compression stats
console.log('\nCompression summary:');
console.log('  Stage 1 (Rust) body:', compressed.length, 'bytes');
console.log('  Stage 2 (RegPack+wrapper):', finalHtml.length, 'bytes');

// Generate QR code
const dataUrl = `data:text/html;charset=utf-8,${finalHtml}`;
console.log('\nData URL length:', dataUrl.length, 'bytes');

try {
    const QRCode = require('qrcode');

    QRCode.toFile(outputQr, dataUrl, {
        width: 800,        // Larger width for max-size QR code
        margin: 1,         // Minimal margin to save space
        errorCorrectionLevel: 'L'  // Low error correction for max data capacity
    }, (err) => {
        if (err) {
            console.error('\nQR generation failed:', err.message);
            console.error('Note: Install qrcode package with: npm install qrcode');
            process.exit(1);
        } else {
            console.log('QR code saved to:', "'" + outputQr + "'");
            console.log('\n✓ Two-stage compression complete!');
        }
    });
} catch (e) {
    console.error('\nQR code generation requires: npm install qrcode');
    console.error('Error:', e.message);
    console.error('\nHTML file was created successfully, but QR code generation failed.');
    process.exit(1);
}
