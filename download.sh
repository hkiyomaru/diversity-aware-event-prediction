#!/usr/bin/env sh

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "${SCRIPT}")

# download datasets
wget "http://nlp.ist.i.kyoto-u.ac.jp/nl-resource/next-event-prediction/data.tar.gz" -O "${SCRIPT_PATH}/data.tar.gz"
tar -xvf "${SCRIPT_PATH}/data.tar.gz"
rm "${SCRIPT_PATH}/data.tar.gz"

# download pre-trained word2vec
mkdir -p "${SCRIPT_PATH}/data/share"
wget "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz" -O "${SCRIPT_PATH}/data/share/GoogleNews-vectors-negative300.bin.gz"
gunzip "${SCRIPT_PATH}/data/share/GoogleNews-vectors-negative300.bin.gz"

# add symlinks
ln -s "${SCRIPT_PATH}/data/share/GoogleNews-vectors-negative300.bin" "${SCRIPT_PATH}/data/descript/source_embedding.bin"
ln -s "${SCRIPT_PATH}/data/share/GoogleNews-vectors-negative300.bin" "${SCRIPT_PATH}/data/descript/target_embedding.bin"
ln -s "${SCRIPT_PATH}/data/share/GoogleNews-vectors-negative300.bin" "${SCRIPT_PATH}/data/wikihow/source_embedding.bin"
ln -s "${SCRIPT_PATH}/data/share/GoogleNews-vectors-negative300.bin" "${SCRIPT_PATH}/data/wikihow/target_embedding.bin"
