#!/usr/bin/env bash
set -e
python tools/train.py --config configs/potsdam.yaml --method suf_hrl
