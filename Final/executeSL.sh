# !/bin/bash

python3 $1 writePhrases spellPhrases publishPhrases $3 > $2/resWrite.txt

python3 $1 providePhrases supplyPhrases leavePhrases $3 > $2/resProvide.txt

python3 $1 meetPhrases visitPhrases satisfyPhrases $3 > $2/resMeet.txt

python3 $1 runPhrases operatePhrases movePhrases $3 > $2/resRun.txt

python3 $1 tryPhrases judgePhrases testPhrases $3 > $2/resTry.txt

python3 $1 buyPhrases purchasePhrases bribePhrases $3 > $2/resBuy.txt

python3 $1 showPhrases expressPhrases picturePhrases $3 > $2/resShow.txt

python3 $1 sayPhrases statePhrases allegePhrases $3 > $2/resSay.txt

python3 $1 acceptPhrases bearPhrases receivePhrases $3 > $2/resAccept.txt

python3 $1 drawPhrases attractPhrases depictPhrases $3 > $2/resDraw.txt

cd $2
cat resWrite.txt resProvide.txt resMeet.txt resRun.txt resTry.txt resBuy.txt resShow.txt resSay.txt resAccept.txt resDraw.txt > resAll.txt

