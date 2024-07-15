#!/bin/bash -e
# A script to create a Lucene index
source ./common_proc.sh
source ./config.sh

checkVarNonEmpty "SAMPLE_COLLECT_ARG"

checkVarNonEmpty "COLLECT_ROOT"
checkVarNonEmpty "LUCENE_INDEX_SUBDIR"
checkVarNonEmpty "INPUT_DATA_SUBDIR"
checkVarNonEmpty "DEFAULT_QUERY_TEXT_FIELD_NAME"

inputDataSubDir="$INPUT_DATA_SUBDIR"

indexSubDir=$LUCENE_INDEX_SUBDIR
indexFieldName="$DEFAULT_QUERY_TEXT_FIELD_NAME"

boolOpts=("h"                     "help"                "print help"
          "exact_match"           "exactMatch"          "create index for exact match"
          "ignore_missing_field"  "ignoreMissingField"  "ignore missing fields (use cautiously, fields normally should not be missing")


paramOpts=(
  "index_field"  "indexFieldName" "indexing field name (default $indexFieldName)"
  "input_subdir" "inputDataSubDir" "input data sub-directory (default $inputDataSubDir)"
  "index_subdir" "indexSubDir"    "index subdirectory (default $indexSubDir)"
)

parseArguments $@

usageMain="<collection>"

if [ "$help" = "1" ] ; then
  genUsage $usageMain
  exit 1
fi

collect=${posArgs[0]}
if [ "$collect" = "" ] ; then
  genUsage "$usageMain" "Specify $SAMPLE_COLLECT_ARG (1st arg)"
  exit
fi

if [ "$exactMatch" = "1" ] ; then
  exactMatchParam=" -exact_match "
else
  exactMatchParam=""
fi

if [ "$ignoreMissingField" = "1" ] ; then
  ignoreMissingFieldParam=" -ignore_missing_field "
else
  ignoreMissingFieldParam=""
fi

inputDataDir="$COLLECT_ROOT/$collect/$inputDataSubDir"
indexDir="$COLLECT_ROOT/$collect/$indexSubDir"

echo "=========================================================================="
echo "Input data directory: $inputDataDir"
echo "Index directory:      $indexDir"
echo "Index field name:     $indexFieldName"
echo "Exact match param:    $exactMatchParam"
echo "Ignore missing field: $ignoreMissingFieldParam"

if [ ! -d "$indexDir" ] ; then
  mkdir -p "$indexDir"
else
  echo "Removing previously created index (if exists)"
  rm -rf "$indexDir"/*
fi
echo "=========================================================================="
retVal=""
getIndexQueryDataDirs "$inputDataDir"
dirList=${retVal[0]}
dataFileName=${retVal[1]}
if [ "$dirList" = "" ] ; then
  echo "Cannot get a list of relevant data directories, did you dump the data?"
  exit 1
fi
if [ "$dataFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
else
  echo "Using the data input file: $dataFileName"
fi
# This APP can be memory greedy
setJavaMem 1 8
LuceneIndexer \
    $exactMatchParam $ignoreMissingFieldParam \
    -input_data_dir "$inputDataDir" \
    -index_dir "$indexDir" \
    -index_field "$indexFieldName" \
    -data_sub_dirs "$dirList" \
    -data_file "$dataFileName"
