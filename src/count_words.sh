#!/bin/sh

# Counts the number of unique words in a file.

echo "the full has number of unique words:";
tr ' ' '\n' < parabank.tsv | sort | uniq -c | wc -l
