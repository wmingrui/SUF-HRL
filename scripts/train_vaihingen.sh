#!/usr/bin/env bash
set -e
python tools/train.py --config configs/vaihingen.yaml --method suf_hrl
