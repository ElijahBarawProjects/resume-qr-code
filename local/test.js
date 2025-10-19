const doRegPack = require('./doRegPack.js');
const content = `\`<c~Q~b$~8www.elijahbaraw.com/~#Elijah Baraw~@~Fa href="mailto:~2?sub~U=RE:%20Your%20Resume~#~2~@ (203) 731-9535 ~A ~8~/~#~/~@</c~Qer><hr>~VEduc~N>Carnegie Mell~^University~cchool of~.~7~(~d~Y1 - May~lFem>B~eelor~M~7~L~.~7. Conc~Qr~NL~.S~Hs</em>~9GPA: 3.97. Relevant courses: ~<, Cloud Comput~T, ~JS~H~E~= ~f~9~%~+ Parallel ~%L~gar Algebra, Diff~bQial Equ~NEIDL~VTechnical Skills~>Langu~_~mC, ~5, Go~cQL, Java, HCL~9~dTechnologie~mNumPy, PyTorch, P~fa~EOpenCV, Linux~c~Ot~EGit, AWS, G~nAzure, K8~ED~Or~9~dTopic~m~=~&~%Ob~U Ori~Q~1Program~T, ~+~9Consensus ~%~BNetwork~Wtocol~ET~nCryptographic ~%~<~VEx~hience~>C~Q~b3Atmospheric P~]Studi~R(~d~A~Fb>Research Assistant~(~dMay~q~Y2~Clow cost device~3measur~-PM2.5 air pollutants collec~u~^a foam tape over several m~ahs~:I~;1an~) pipel~g as a chea~h alternative~4tradi~'al p~]~Z~' m~ein~R:Integra~ua~iH~MArduino~&~5 scripts~4c~arol step~h motors bas~1~^c~ainuous CV~Lput.</ul>~VPro~Us~>F~aify (Solo ~5~W~U)~6PIL, Im~_~Wce~jT, De-Nois~T,~.Vision~*Jan~q~r2~"Created~) soft~s ~5~4convert h~fwritten letters~Lto a ~hsonaliz~1bitmap f~a~k~ted~), edge-~Z~onoise reduc~' algorithms~4~Zt pencil writ~-~^pa~h.~?C~0~Wxy S~I (C)~6Git, HTTP~c~Ots~*~r3~Cproxy~iI~L C~[p_threads~&fork~4h~fle~Pc~0ly, anonymize traffic~&c~ee responses. ~~uUnix~iOts~&a bound~1c~ee fo~van LRU evic~' policy.~?~JBackend (Golang)~6Replica~o~BMailbox/Me~j_ Pa~jT~*Nov~,3~"Designed~&~wua c~0~iI~4man~_ ~X~x3a~yplayer game, accessible via API~kH~fl~1cli~Q~Pabout~&updat~R4~Xgame ~x[RPCs~&a me~j_-pa~j-model~kI~;1node launch~T~&s~I~z~Eensur~-replic~N&enforc~-consistency within~zs.~?~Ss Hackath~^Dev Team~6G~nK8~E~D*Mar~,4; Mar~l"~{CMU Data ~7 Club run ~!first AI ~S ~G~owith $6,000~L priz~R&63 teams~kUs~1~D4automatically ~}D~Or imag~RMuser-submit~u~5 bot~Ea~v~Gtors~4use custom dependenci~R&m~e~g learn~-librari~RM~!choice, runn~-c~aa~grs ~^GCP~k~{~}~Xsecond it~bNM~X~G~'~L~l[AWS ECS~3bot~ELambda~3m~p~R?x86 IA-32 K~KScr~p~6C, ASM~cimics~*~Y4 - Dec~,4~"Built a complete i386 k~Kscr~p, solo,~3CMU 15410, i~;-preemptive~ytask~T~:Eng~g~b1hard~sterface~Ememory man~_m~Q,~&I/O~iH~3c~0 ELF binary ~w'</ul>\``;
const word_list = `\`build |their |Help~1| group| multi|state~|execu~|llow~-|t~1|Utiliz|ware~L|July~,|~,2 - |atch|~', |CP, |s:~(|~,5~|.~:|ss~| s~|per|ine|and|ach|<b>|, S|er~|ont||age|on |article || us~-|detec|Aug~,|the | Pro|<h1>|ject|ing|Poker-Bot|es~|ent| requests |ocke|a~'~| of | in|ernel from |Distribut~1|erver|ystem|competi|</b>~9<|s, |GitHub Ac~'s~|~"Develop~1a |Actor Model, |Pittsburgh, PA|</u></a>|</ul~$|</h1~$|Data Structures|Machine Learning|mplement~|<li>|<br>|<a href="https://|Science|~(<em>|Python| to | for |ebaraw@andrew.cmu.edu|ed |oncurrent|github.com/elijah-bae-raw| Computer |ing | 202|Func~'al Programming, Systems,|</em> <b>| image processing|</b> |tion| and |Algorithms, |><p><b>|" style="color:#00f"><u>|</b><ul><li>\``

function byteLen(inString) {
    return encoded_len = encodeURI(inString).length;
}

var output = doRegPack(content);
if (output.endsWith('eval(_)')) {
    output = output.slice(0, -7)
}
output = output.replaceAll("\\`", "")



function toHexEscape(str) {
    return str.replace(/[\u0001-\u001f]/g, char => {
        const code = char.charCodeAt(0);
        return '\\x' + code.toString(16).padStart(2, '0');
    });
}

console.log("In: \t", content)
// console.log("Out:\t", `decodeURIComponent("${encodeURI(output)}")`);
console.log("Out:\t", toHexEscape(output));
console.log("Eval:\t", eval(output));
console.log(`Input Bytes: ${content.length}; Output Bytes: ${output.length}; savings: ${content.length - output.length} (positive is good)`)

// Write the raw output to a JavaScript file
const fs = require('fs');

fs.writeFileSync('./test.js.out.js', output);

const solution = `<meta charset="utf-8">
<script>
    onload = () => {
        let w = ${word_list}.split("|")
        ${output}
        for (i = 0; i < 92;)_ = _.replaceAll("~" + String.fromCharCode(125 - i), w[i++]); document.body.innerHTML = _
    };
</script>`
fs.writeFileSync('./resume.regpat.html', solution)

// Create data URL for the compressed HTML with UTF-8 encoding
const dataUrl = `data:text/html;charset=utf-8,${solution}`;
console.log('\nData URL length:', dataUrl.length);
console.log('Data URL preview:', dataUrl.substring(0, 100) + '...');

// Generate QR code with maximum capacity settings
try {
    const QRCode = require('qrcode');
    QRCode.toFile('./resume-qr.png', dataUrl, {
        width: 800,        // Larger width for max-size QR code
        margin: 1,         // Minimal margin to save space
        errorCorrectionLevel: 'L'  // Low error correction for max data capacity
    }, (err) => {
        if (err) console.error('QR generation failed:', err.message);
        else console.log('QR code saved as resume-qr.png');
    });
} catch (e) {
    console.log('QR code generation requires: npm install qrcode');
}