#!/usr/bin/env sh
set -ex
shopt -s extglob
ZIP_FILE=code.zip
rm $ZIP_FILE
zip $ZIP_FILE !(CollabFilteringEvaluation).m
shopt -u extglob
