#!/bin/sh
(
    echo "Create Version Info"
    sh vc -f -m
)
TMPDIR=$(mktemp -d -p ".")
echo "Using temporary directory: $TMPDIR"
echo "Create TWBooks"
pdflatex.exe twbook.ins
echo "Erstellen der Dokumentation"
pdflatex.exe -draftmode -interaction=batchmode -aux-directory="$TMPDIR" -output-directory="." twbook.dtx
makeindex.exe -s gind.ist -o "$TMPDIR/twbook.ind" "$TMPDIR/twbook.idx"
makeindex.exe -s gglo.ist -o "$TMPDIR/twbook.gls" "$TMPDIR/twbook.glo"
pdflatex.exe -draftmode -interaction=batchmode -aux-directory="$TMPDIR" -output-directory="." twbook.dtx
pdflatex.exe -interaction=batchmode -aux-directory="$TMPDIR" -output-directory="." twbook.dtx
echo "Erstellen des Beispieles"
pdflatex.exe -draftmode -interaction=batchmode -aux-directory="$TMPDIR" -output-directory="." BSP_MA_Code_BibTeX.tex
bibtex.exe  "$TMPDIR/BSP_MA_Code_BibTeX.aux"
pdflatex.exe -draftmode -interaction=batchmode -aux-directory="$TMPDIR" -output-directory="."  BSP_MA_Code_BibTeX.tex
pdflatex.exe -interaction=batchmode -aux-directory="$TMPDIR" -output-directory="."  BSP_MA_Code_BibTeX.tex
echo "Remove temporary directory"
rm -rf "$TMPDIR"
rm "twbook.log"
