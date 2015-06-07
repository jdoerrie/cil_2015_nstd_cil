#!/usr/bin/env sh
set -x
shopt -s extglob
ZIP_FILE=code.zip
cd $1
rm $ZIP_FILE
zip $ZIP_FILE !(CollabFilteringEvaluation).m
shopt -u extglob
