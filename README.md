# yolox-torchscript-python

## YOLOX
https://github.com/Megvii-BaseDetection/YOLOX

## Torchscriptへの変換
YOLOXで提供されているconvert_torchscript.pyを使用。yolox-tiny.torchscript.ptを生成。
```
bash convert_torchscript.sh
```

## Torchscriptで推論
```
python3 yolox_torchscript.py
```

## サンプル画像

サンプル画像は下記の映像から切り出して使用させていただきました。

https://www2.nhk.or.jp/archives/movies/?id=D0002011239_00000
