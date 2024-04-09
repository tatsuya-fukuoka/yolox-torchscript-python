# yolox-torchscript-python

## YOLOX
https://github.com/Megvii-BaseDetection/YOLOX

## Torchscriptへの変換
YOLOXで提供されているconvert_torchscript.pyを使用。yolox-tiny.torchscript.ptを生成。
```
python tools/export_torchscript.py --output-name yolox-tiny.torchscript.pt -n yolox-tiny -c yolox_tiny.pth --decode_in_inference
```

## Torchscriptで推論
```
python yolox_torchscript.py
```