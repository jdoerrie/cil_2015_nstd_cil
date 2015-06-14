#!/usr/bin/env bash

if [[ $1 == '' ]]; then
  echo "Usage: ${0} Grid_Search.txt [EPOCH]"
  exit
fi

FILE=$1
if [[ $2 == '' ]]; then
  RMSE=$(grep "RMSE =" "${FILE}")
  SORTED=$(echo "$RMSE" | sort -t',' -rk6)
  BEST=$(echo "$SORTED" | tail -1)
  echo "Best Overall Result:"
  grep -B1 "$BEST" "$FILE"
else
  EPOCH=$(printf "%03d" "$2")
  EPOCHS=$(grep "Epoch: ${EPOCH}" "${FILE}")
  SORTED=$(echo "$EPOCHS" | sort -t':' -rk2)
  BEST=$(echo "$SORTED" | tail -1)
  BEST_SCORE=$(echo "$BEST" | grep -oE "0.9[0-9]+")
  NEXT_LINES=$(grep -A 500 "Epoch: $EPOCH, Curr RMSE: $BEST_SCORE" "$FILE")
  PARAMS=$(echo "$NEXT_LINES" | grep -m1 -B1 "RMSE = ")
  echo "Best Result after ${EPOCH} Epochs:"
  echo "$BEST"
  echo "$PARAMS"
fi
