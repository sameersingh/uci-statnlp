#! /bin/bash

# Make data directory
mkdir -p data/
cd data/

# CoNLL-U data for POS tagging
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English/master/en-ud-dev.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English/master/en-ud-test.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English/master/en-ud-train.conllu

# Movie Review Dataset for sentiment classification
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzvf aclImdb_v1.tar.gz
rm aclImdb_v1.tar.gz

# Shakespeare
wget http://norvig.com/ngrams/shakespeare.txt

cd ..
