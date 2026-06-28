#!/usr/bin/env bash
set -e
python tools/train.py --config configs/loveda.yaml --method suf_hrl
