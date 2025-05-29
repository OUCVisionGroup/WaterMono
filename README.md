<div id="top" align="center">
  
# WaterMono 
**Teacher-Guided Anomaly Masking and Enhancement Boosting for Robust Underwater Self-Supervised Monocular Depth Estimation**
  [[paper link]](https://ieeexplore.ieee.org/abstract/document/10937915)
  
  Yilin Ding, Kunqian Li*, Han Mei, Shuaixin Liu and Guojia Hou
  
<img src="./assets/sample.gif" width="50%" alt="teaser" align=center />

(WaterMono 448x288) 
  
</div>

## ⚙️ Setup

You can install the dependencies with:
```shell
pip install -r requirements.txt
pip install 'git+https://github.com/saadnaeem-dev/pytorch-linear-warmup-cosine-annealing-warm-restarts-weight-decay'
```
We ran our experiments with PyTorch 1.12.1, CUDA 11.7, Python 3.9.18 and Ubuntu 18.04.
 If you encounter problems with the environment setup, you can refer to [Lite-Mono](https://github.com/noahzn/Lite-Mono)'s README file and issues.

## 💾 Data Preparation
**Dataset**

We mainly used the FLSea dataset. You can download the FLSea-VI dataset from [here](https://www.kaggle.com/datasets/viseaonlab/flsea-vi) 
and the FLSea-stereo dataset from [here](https://www.kaggle.com/datasets/viseaonlab/flsea-stereo). 
 As for our manually created challenge dataset for measuring rotation robustness, it can be downloaded from [here](https://drive.google.com/file/d/1C_r4OYnqXVy0gCnSiAfq-MxAPI7xVesz/view?usp=sharing).

Our default settings expect that you have converted the tiff images to jpeg to save memory during training. You can convert the format using the following command, which also deletes the raw FLSea `.tiff` files.

    find archive/ -name '*.tiff' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.tiff {.}.jpg && rm {}'

**Splits**

The train/test/validation splits are defined in the `splits/` folder.
Our proposed split method is referred to as the `OUC_split`. 
You can also define your own split method and use them by setting the `--split` flag.

**Enhanced Images**

The enhanced images should be placed in `/FLSea/location/scene/scene/IEB`. You may refer to the image enhancement code in [Sea-thru](https://github.com/hainh/sea-thru) or use the pre-enhanced images provided in the FLSea dataset.


## 📦 Models

| Name                       | Input size | OUC disparities                       |
|----------------------------|------------|--------------------------------------|
| Lite-Mono(Teacher Network) | 448 x 288  | [Download 🔗](https://drive.google.com/drive/folders/16MyoFIiVTm34hUq50YGhKV2sZa0jB3o8?usp=sharing)|
| WaterMono(Student Network) | 448 x 288  | [Download 🔗](https://drive.google.com/drive/folders/1VuTUXKWjytGWsysmnU9qPrGErGcHtRNS?usp=sharing) |




## 📊 Test and Evaluation
**Test**

You can predict disparity for a single image with:

    python test_simple.py --load_weights_folder path/to/your/weights/folder --image_path path/to/your/test/image

**Evaluation**

If you want to evaluate the model on the test set defined by `OUC_split`, first prepare the ground truth depth maps by running:

    python export_gt_depth.py

Then evaluate the model by running:

    python evaluate_depth.py --load_weights_folder path/to/your/weights/folder --data_path path/to/FLSea_data/ --model lite-mono

If you want to test generalization on the FLSea-stereo dataset, please add flag `--eval_stereo`.

## 🕒Training
#### start training
    python train.py --data_path path/to/your/data --model_name mytrain --num_epochs 30 --batch_size 12
    
#### tensorboard visualization
    tensorboard --log_dir ./tmp/mytrain

## 💕Thanks
Our code is based on [Monodepth2](https://github.com/nianticlabs/monodepth2), [Lite-Mono](https://github.com/noahzn/Lite-Mono) and [Sea-thru](https://github.com/hainh/sea-thru). You can refer to their README files and source code for more implementation details. 

## 🖇️Citation

  ```bibtex
  @article{ding2025watermono,
  title={WaterMono: Teacher-guided anomaly masking and enhancement boosting for robust underwater self-supervised monocular depth estimation},
  author={Ding, Yilin and Li, Kunqian and Mei, Han and Liu, Shuaixin and Hou, Guojia},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2025},
  publisher={IEEE}
  }
