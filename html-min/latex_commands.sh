#!/bin/bash
# usage: invoke with first arugment pointing to the .tex resume file
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resume_path="$SCRIPT_DIR/out/resume.html"
resume_min_path="$SCRIPT_DIR/out/resume.min.html"
template_path="$SCRIPT_DIR/templates/minimal.html"
tex_path=$1
python_dir="$SCRIPT_DIR/compress"

# make
pandoc -f latex-auto_identifiers $tex_path  -o "$resume_path" \
  --standalone \
  --no-highlight \
  --metadata title="" \
  --template="$template_path" \
  --strip-comments
echo "Made. Size: $(wc -c < "$resume_path")"

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
  "$resume_path"
echo "Tidied. Size: $(wc -c < "$resume_path")"


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
  "$resume_path" -o "$resume_path"

echo "Minified. Size: $(wc -c < "$resume_path")"


# hand-crafted substitutions
sed -i '' 's/<strong>/<b>/g; s/<\/strong>/<\/b>/g' "$resume_path"
sed -i '' 's/<emphasis>/<i>/g; s/<\/emphasis>/<\/i>/g' "$resume_path"
sed -i '' 's/<paragraph>/<p>/g; s/<\/paragraph>/<\/p>/g' "$resume_path"
sed -i '' 's/–/-/g' "$resume_path"
sed -i '' 's/<span><b>\([^<]*\)<\/b><\/span>/<b>\1<\/b>/g' "$resume_path"
sed -i '' 's/<span><i>\([^<]*\)<\/i><\/span>/<i>\1<\/i>/g' "$resume_path"
sed -i '' 's/<span><u>\([^<]*\)<\/u><\/span>/<u>\1<\/u>/g' "$resume_path"
sed -i '' 's/<span><em>\([^<]*\)<\/em><\/span>/<em>\1<\/em>/g' "$resume_path"
sed -i '' 's/<title><\/title>//g' "$resume_path"
sed -i '' 's/<style><\/style>//g' "$resume_path"
sed -i '' 's/<span><b><span style="color:#00f"><u><a href="\([^"]*\)">\([^<]*\)<\/a><\/u><\/span><\/b><\/span>/<b><a href="\1" style="color:#00f"><u>\2<\/u><\/a><\/b>/g' "$resume_path"
sed -i '' 's/<span><span style="color:#00f"><u><a href="\([^"]*\)">\([^<]*\)<\/a><\/u><\/span>\([^<]*\)<span style="color:#00f"><u><a href="\([^"]*\)">\([^<]*\)<\/a><\/u><\/span><\/span>/<a href="\1" style="color:#00f"><u>\2<\/u><\/a>\3<a href="\4" style="color:#00f"><u>\5<\/u><\/a>/g' "$resume_path"
sed -i '' 's/<li><p>/<li>/g' "$resume_path" # note: this has a visual change of removing spacing between bullet points
sed -i '' 's/<div class="center">/<center>/g; s/<\/div>/<\/center>/g' "$resume_path" # note: this is deprecated in html5 but works (24 Aug 2025) on Safari / Firefox

echo "Custom minify. Size: $(wc -c < "$resume_path")"

# Custom JS based compression
python3 "$python_dir/optimize_compression.py" --input="$resume_path" --output="$resume_min_path" && echo "JS Based Approach. Size: $(wc -c < "$resume_min_path")"
