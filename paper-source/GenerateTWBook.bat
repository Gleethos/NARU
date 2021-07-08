@echo off
REM Batch-Datei zur Erstellung der Klasse TWBook und ihrer Dokumentation
echo XXX Batch-Datei zum erstellen der Klasse TWBook
if exist "C:\gnuwin32\bin\gawk.exe" (
	echo GAWK vorhanden
	CALL vc.bat -f -m
) else (
	echo Bitte die GNUWin32-Utilities unter "C:\gnuwin32" mit GAWK installieren!
	pause
)
REM Erstellen eines Verzeichnisses für die Hilfsdateien
mkdir temp
REM Erstellen der Datei twbook.cls
echo XXX Erstellen der Klasse twbook
pdflatex twbook.ins
REM Erstellen der Dokumentation
echo XXX Erstellen der Dokumentation
pdflatex -draftmode -interaction=batchmode -aux-directory=".\temp" -output-directory="." twbook.dtx
makeindex -s gind.ist -o "temp\twbook.ind" "temp\twbook.idx"
makeindex -s gglo.ist -o "temp\twbook.gls" "temp\twbook.glo"
pdflatex -draftmode -interaction=batchmode -aux-directory=".\temp" -output-directory="." twbook.dtx
pdflatex -interaction=batchmode -aux-directory=".\temp" -output-directory="." twbook.dtx
REM Erstellen des Beispiels
echo XXX Erstellen des Beispiels BSP_MA_Code_BibTeX
pdflatex -draftmode -interaction=batchmode -aux-directory=".\temp" -output-directory="." BSP_MA_Code_BibTeX.tex
bibtex  temp\BSP_MA_Code_BibTeX.aux
pdflatex -draftmode -interaction=batchmode -aux-directory=".\temp" -output-directory="."  BSP_MA_Code_BibTeX.tex
pdflatex -interaction=batchmode -aux-directory=".\temp" -output-directory="."  BSP_MA_Code_BibTeX.tex
pause
DEL twbook.log
RMDIR /S /Q temp
