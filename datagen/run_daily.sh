#!/bin/sh
# Daily own-data generation (launchd: ~/Library/LaunchAgents/com.ntl.datagen.plist). Logs to datagen/logs/.
cd /Users/abhayganti/name-that-llm-kaggle || exit 1
mkdir -p datagen/logs
./.venv/bin/python datagen/daily.py --prompts 40 >> "datagen/logs/$(date -u +%Y%m%d).log" 2>&1
