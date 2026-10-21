#!/bin/sh
# usage: make.sh <slug> <hidden> <n_layers> <keep> <stride> '<model_sources json list>' '<dataset_sources json list>'
d=feat-$1; mkdir -p $d
sed -e "s/__MODEL_TYPE__/any/" -e "s/__HIDDEN__/$2/" -e "s/__NLAYERS__/$3/" -e "s/__KEEP__/$4/" -e "s/__STRIDE__/$5/" -e "s/__TAG__/$1/" feat_kernel.py > $d/main.py
cat > $d/kernel-metadata.json <<JSON
{"id": "abhayganti09/ntl-feat-$1", "title": "ntl-feat-$1", "code_file": "main.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": false,
 "dataset_sources": $7, "competition_sources": ["name-that-llm"], "kernel_sources": [],
 "model_sources": $6, "machine_shape": "NvidiaTeslaT4"}
JSON
echo $d
