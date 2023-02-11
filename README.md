# NeRFの実装練習

## 参考資料
https://blog.albert2005.co.jp/2020/05/08/nerf/

## Dockerコンテナ生成コマンド
```bash
docker run --gpus all -it --name nerf_practice_container nvcr.io/nvidia/pytorch:22.12-py3 bash
```

## 使用データ
https://www.vincentsitzmann.com/deepvoxels/

より
`synthetic_scenes.zip`
を`data/`以下に展開する

```
nerf_practice/
  |--data/
    |--test/
    |--train/
    |--validation/
```
となっている想定

## 環境構築
libtorchの導入
```bash
./shell_script/download_libtorch.sh 
```

OpenCVの導入
```bash
sudo apt install libopencv-dev
```

Eigenの導入
```bash
sudo apt install libeigen3-dev -y
```
