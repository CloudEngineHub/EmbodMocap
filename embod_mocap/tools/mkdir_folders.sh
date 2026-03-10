#!/bin/bash

num_folders=$1   # first parameter: number of folders
output_dir=$2    # second parameter: output folder path


# create specified number of folders
for i in $(seq 0 $((num_folders - 1))); do
  folder_name="seq$i"
  folder_path="$output_dir/$folder_name"
  mkdir -p "$folder_path"
done

echo "all $num_folders folders created to $output_dir"