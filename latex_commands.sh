# make
pandoc -f latex-auto_identifiers Elijah.Baraw.Resume.tex  -o resume.html \
  --standalone \
  --no-highlight \
  --metadata title="" \
  --template=minimal.html \
  --strip-comments
echo "Made. Size: $(wc -c resume.html)"

# tidy
tidy -m \
  -w 0 \
  -omit \
  --drop-empty-elements yes \
  --drop-empty-paras yes \
  --join-styles yes \
  --merge-divs auto \
  --merge-emphasis yes \
  --merge-spans yes \
  --omit-optional-tags yes \
  --replace-color yes \
  --quiet yes \
  --tidy-mark no \
  --fix-uri no \
  resume.html
echo "Tidied. Size: $(wc -c resume.html)"


# minify
html-minifier-terser \
  --collapse-whitespace \
  --remove-comments \
  --remove-optional-tags \
  --remove-redundant-attributes \
  --remove-script-type-attributes \
  --remove-tag-whitespace \
  --use-short-doctype \
  --minify-css true \
  --minify-js true \
  resume.html -o resume.html

echo "Minified. Size: $(wc -c resume.html)"


# hand-crafted substitutions
sed -i '' 's/<strong>/<b>/g; s/<\/strong>/<\/b>/g' resume.html
sed -i '' 's/<emphasis>/<i>/g; s/<\/emphasis>/<\/i>/g' resume.html
sed -i '' 's/<paragraph>/<p>/g; s/<\/paragraph>/<\/p>/g' resume.html
sed -i '' 's/–/-/g' resume.html
sed -i '' 's/<span><b>\([^<]*\)<\/b><\/span>/<b>\1<\/b>/g' resume.html
sed -i '' 's/<span><i>\([^<]*\)<\/i><\/span>/<i>\1<\/i>/g' resume.html
sed -i '' 's/<span><u>\([^<]*\)<\/u><\/span>/<u>\1<\/u>/g' resume.html
sed -i '' 's/<span><em>\([^<]*\)<\/em><\/span>/<em>\1<\/em>/g' resume.html
sed -i '' 's/<title><\/title>//g' resume.html
sed -i '' 's/<style><\/style>//g' resume.html
sed -i '' 's/<span><b><span style="color:#00f"><u><a href="\([^"]*\)">\([^<]*\)<\/a><\/u><\/span><\/b><\/span>/<b><a href="\1" style="color:#00f"><u>\2<\/u><\/a><\/b>/g' resume.html
sed -i '' 's/<span><span style="color:#00f"><u><a href="\([^"]*\)">\([^<]*\)<\/a><\/u><\/span>\([^<]*\)<span style="color:#00f"><u><a href="\([^"]*\)">\([^<]*\)<\/a><\/u><\/span><\/span>/<a href="\1" style="color:#00f"><u>\2<\/u><\/a>\3<a href="\4" style="color:#00f"><u>\5<\/u><\/a>/g' resume.html
sed -i '' 's/<li><p>/<li>/g' resume.html # note: this has a visual change of removing spacing between bullet points
sed -i '' 's/<div class="center">/<center>/g; s/<\/div>/<\/center>/g' resume.html # note: this is deprecated in html5 but works (24 Aug 2025) on Safari / Firefox

echo "Custom minify. Size: $(wc -c resume.html)"

# Custom JS based compression
python3 optimize_compression.py --input=resume.html --output=resume.min.html && echo "JS Based Approach. Size: $(wc -c resume.min.html)"


# # Add script after word replacements to avoid corrupting script content
# sed -i '' 's/<meta charset="utf-8">/<script>onload=()=>{let w=`GitHub |ont|age|on |article | us~l|detec|Aug~k|the | Pro|<h1>|ject|ing|ent| requests |ocke|a~f~|es~| of | in|ernel from |Distribut~p|erver|ystem|competi|<\/b>~x<|s, |~aDevelop~pa |Actor Model, |Pittsburgh, PA|<\/u><\/a>|<\/ul~c|<\/h1~c|Data Structures|Machine Learning|mplement~|<li>|<br>|<a href="https:\/\/|Science|~g<em>|Python| to | for |ebaraw@andrew.cmu.edu|ed |oncurrent|github.com\/elijah-bae-raw| Computer |ing | 202|Func~fal Programming, Systems,|<\/em> <b>| image processing|<\/b> |tion| and |Algorithms, |><p><b>|" style="color:#00f"><u>|<\/b><ul><li>`.split("|"),h=document.body.innerHTML,i=0;for(c of"876543210ZYXWVUTSRQPONMLKJIHGFEDCBAzyxwvutsrqponmlkjihgfedcba")h=h.replaceAll("~"+c,w[i++]);document.body.innerHTML=h};<\/script><meta charset="utf-8">/g' resume.html
# script='<script>onload=()=>{let w=`GitHub |ont|age|on |article | us~l|detec|Aug~k|the | Pro|<h1>|ject|ing|ent| requests |ocke|a~f~|es~| of | in|ernel from |Distribut~p|erver|ystem|competi|<\/b>~x<|s, |~aDevelop~pa |Actor Model, |Pittsburgh, PA|<\/u><\/a>|<\/ul~c|<\/h1~c|Data Structures|Machine Learning|mplement~|<li>|<br>|<a href="https:\/\/|Science|~g<em>|Python| to | for |ebaraw@andrew.cmu.edu|ed |oncurrent|github.com\/elijah-bae-raw| Computer |ing | 202|Func~fal Programming, Systems,|<\/em> <b>| image processing|<\/b> |tion| and |Algorithms, |><p><b>|" style="color:#00f"><u>|<\/b><ul><li>`.split("|"),h=`<c~Ver~c~wwww.elijahbaraw.com\/~bElijah Baraw~E~Ja href="mailto:~q?sub~X=RE:%20Your%20Resume~b~q~E (203) 731-9535 ~F ~w~n~b~n~E<\/c~Ver><hr>~YEduc~SCCarnegie Mell~5University, School of~m~v~g<b>~11 - May~k5~Jem>Bachelor~Q~v~P~m~v. Conc~Vr~SP~mS~Ls<\/em>~xGPA: 3.97. Relevant courses: ~A, Cloud Comput~W, ~NS~L~I~B and~x~d~j Parallel ~dLinear Algebra, Differ~Vial Equ~SIIDL~YTechnical Skills~CLangu~6s:~gC, ~t, Go, SQL, Java, HCL~x<b>Technologies:~gNumPy, PyTorch, Panda~IOpenCV, Linux, S~Tt~IGit, AWS, GCP, Azure, K8~ID~Tr~x<b>Topics:~g~B~e~dOb~X Ori~V~pProgram~W, ~j~xConsensus ~d~GNetwork~Ztocol~ITCP, Cryptographic ~d~A~YExperience~CC~Ver~rAtmospheric P~4Studi~Rg<b>~F~Jb>Research Assistant~g<b>May~k2 - ~12~Hlow cost device~rmeasur~lPM2.5 air pollutants collect~p~5a foam tape over several m~7hs~yI~zpan~h pipeline as a cheaper alternative~stradi~fal p~4~2~f machin~RyIntegrat~pa s~L~QArduino~e~t scripts~sc~7rol stepper motors bas~p~5c~7inuous CV~Pput.<\/ul>~YPro~Xs~CF~7ify (Solo ~t~Z~X)~uPIL, Im~6~Zcess~W, De-Nois~W,~mVision~iJan~k2 - July~k2~aCreated~h software~P ~t~sconvert handwritten letters~Pto a personaliz~pbitmap f~7.~yUtilized~h, edge-~2~f, noise reduc~f algorithms~s~2t pencil writ~l~5paper.~DC~o~Zxy S~M (C)~uGit, HTTP, S~Tts~iJuly~k3~Hproxy s~M~P C~3p_threads~efork~shandle~Uc~oly, anonymize traffic~ecache responses. Utiliz~pUnix s~Tts~ea bound~pcache follow~lan LRU evic~f policy.~D~NBackend (Golang)~uReplica~f, ~GMailbox\/Mess~6 Pass~W~iNov~k3~aDesigned~eexecut~pa c~o s~M~sman~6 ~0state~ra multiplayer game, accessible via API.~yHandl~pcli~V~Uabout~eupdat~Rs~0game state~3RPCs~ea mess~6-pass~lmodel.~yI~zpnode launch~W~~RM group~Iensur~lreplic~Seenforc~lconsistency within groups.~DPoker-Bots Hackath~5Dev Team~uGCP, K8~I~8Ac~fs~iMar~k4; Mar~k5~aHelp~pCMU Data ~v Club run their first AI Poker bot ~K~f, with $6,000~P priz~Re63 teams.~yUs~p~8ac~fs~sautomatically build d~Tr imag~RQuser-submitt~p~t bot~Iallow~l~Ktors~suse custom dependenci~Remachine learn~llibrari~RQtheir choice, runn~lc~7ainers ~5GCP.~yHelp~pbuild ~0second iter~SQ~0~K~f~P~k5~3AWS ECS~rbot~ILambda~rmatch~RDx86 IA-32 K~OScratch~uC, ASM, Simics~i~14 - Dec~k4~aBuilt a complete x86-32 k~Oscratch, solo,~rCMU 15410, i~zlpreemptive multitask~W~yEngineer~phardware~Pterface~Imemory man~6m~V,~eI\/O s~L~rc~o ELF binary execu~f<\/ul>`,i=0;for(c of"876543210ZYXWVUTSRQPONMLKJIHGFEDCBAzyxwvutsrqponmlkjihgfedcba")h=h.replaceAll("~"+c,w[i++]);document.body.innerHTML=h};</script>'
# html="<meta charset="utf-8">$script"
# echo $html > resume.html
# echo "Custom JS-based substitution. Size: $(wc -c resume.html)"

