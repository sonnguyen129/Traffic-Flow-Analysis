# Trafic-Management-System

![Python](https://img.shields.io/badge/Python-3.8.10-green.svg)
![Pytorch](https://img.shields.io/badge/Pytorch-1.12.0+cu102-red.svg)
![YOLO](https://img.shields.io/badge/Model-YOLOX-yellow.svg)
![Tracking](https://img.shields.io/badge/Tracking-DeepSORT-blueviolet.svg)
![Dashboard](https://img.shields.io/badge/Dashboard-Plotly-important.svg)
![Frontend](https://img.shields.io/badge/Framework-Flask-ff69b4.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/sonnguyen129/Trafic-Management-System/graphs/commit-activity))

## Web Application: 
Built a web application using Flask and Dash.

![demo](./assets/webapp-demo.gif)

## Installation

* Clone this repository and check the ```requirements.txt```:
    ```shell
    git clone https://github.com/sonnguyen129/Trafic-Management-System
    cd Trafic-Management-System
    pip install -r requirements.txt
    ```

* Download YOLOX-S,M,L weights in [this repository](https://github.com/Megvii-BaseDetection/YOLOX) and add in ```./weights``` folder

* Simply run Deploy webapp notebook.

    **Note**: The reason I don't deploy this repository to *Heroku* because it's very heavy, runs **1-2 FPS** with CPU. So I used *ngrok* to be able to deploy online and used Colab to take advantage of the GPU. However, run it if you have a powerful enough GPU (Colab's GPU runs around **8-14 FPS**, quite slow but better than CPU)


## 



## References
* [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/pdf/2107.08430.pdf)
* [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/pdf/1703.07402.pdf)
* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [DeepSORT](https://github.com/ZQPei/deep_sort_pytorch)