#!/bin/bash

# binフォルダが存在しない場合は作成
mkdir -p bin

echo "[ビルド開始] コンパイルを行っています..."

# g++を使って、CファイルとC++ファイルを一気にコンパイルし、binフォルダに出力
g++ -O3 -fopenmp src/scaling.c src/ai_scale.cpp \
    -o bin/upscale_app \
    $(pkg-config --cflags --libs opencv4) \
    -lopenvino -lm

# 実行結果の確認
if [ $? -eq 0 ]; then
    echo "[ビルド成功] 実行ファイルを作成しました: bin/upscale_app"
else
    echo "[ビルド失敗] エラー内容を確認してください。"
fi
