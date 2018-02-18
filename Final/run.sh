# !/bin/bash

echo "model selectional preference, dot product, word2vec"

mkdir ResDotW2V

./executeSL.sh embeddingW2V.py ResDotW2V dot

echo "model selectional preference, multiplication, word2vec"

mkdir ResMalW2V

./executeSL.sh embeddingW2V.py ResMalW2V multiplication

echo "model selectional preference, dot product, GloVe"

mkdir ResDotGlove

./executeSL.sh embeddingGlove.py ResDotGlove dot

echo "model selectional preference, multiplication, GloVe"

mkdir ResMalGlove

./executeSL.sh embeddingGlove.py ResMalGlove multiplication

echo "model baseline, addition, word2vec"

mkdir ResBaseAddW2V

./executeBaseLine.sh baselineW2V.py ResBaseAddW2V add

echo "model baseline, multiplication, word2vec"

mkdir ResBaseMalW2V

./executeBaseLine.sh baselineW2V.py ResBaseMalW2V mal

echo "model baseline, addition, GloVe"

mkdir ResBaseAddGlove

./executeBaseLine.sh baselineGlove.py ResBaseAddGlove add

echo "model baseline, multiplication, GloVe"

mkdir ResBaseMalGlove

./executeBaseLine.sh baselineGlove.py ResBaseMalGlove mal

python resultCompute.py > result.txt




