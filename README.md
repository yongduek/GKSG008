# GKSG008/KOS5008 

Files used for GKSG008/KOS5008 Programming for Deep Learning.

# Drawing/Cartoon/Anime GAN
1. [Generative Adversarial Networks for photo to Hayao Miyazaki style cartoons, 2020](https://arxiv.org/abs/2005.07702)
    - [Cartoon-gan pytorch github by Filip Andersson](https://github.com/FilipAndersson245/cartoon-gan)
1. [CartoonGAN: Generative Adversarial Networks for Photo Cartoonization, 2018, github](https://github.com/znxlwm/pytorch-CartoonGAN)
    - [another pytorch in github](https://github.com/TobiasSunderdiek/cartoon-gan)
1. [AnimeGANv2, the improved version of AnimeGAN.](https://github.com/TachibanaYoshino/AnimeGANv2)
    - [pytorch version](https://github.com/bryandlee/animegan2-pytorch)
1. [Anime Face Drawing by GAN]
    - [github](https://github.com/jayleicn/animeGAN)
    - [another github](https://github.com/nikitaa30/Manga-GAN)
1. [Line Drawings for Face Portraits from Photos using Global and Local Structure based GANs, 2020](https://github.com/yiranran/APDrawingGAN2)
1. [ArtLine Drawing](https://github.com/vijishmadhavan/ArtLine)

### Related
  1. [Sketch-based deep learning papers](https://github.com/qyzdao/Sketch-Based-Deep-Learning)
  1. [GAN artwork generation](https://github.com/otepencelik/GAN-Artwork-Generation)
  1. [AI Art, github](https://github.com/Adi-iitd/AI-Art)
        - Neural style transfer, Pix2Pix, CycleGAN

# Document Analysis/Understanding
1. [ICDAR 2019 Tutorial on Deep learning for OCR, Doc Analysis, Text Recog., & Language Modeling](https://github.com/tmbdev-tutorials/icdar2019-tutorial)
2. [Tesseract-OCR, C++](https://github.com/tesseract-ocr/tesseract)
3. [MMOCR,pytorch 1.6+](https://github.com/open-mmlab/mmocr)
    - Full pytorch 
5. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
    - based on paddle library, similar to pytorch
7. [Awesome Doc. Understanding](https://github.com/tstanislawek/awesome-document-understanding)
8. [Deep text recognition benchmark, by Clovaai, pytorch, github](https://github.com/clovaai/deep-text-recognition-benchmark)

# Font
1. [Zi2Zi, pytorch](https://github.com/EuphoriaYan/zi2zi-pytorch)
1. [Multi-content GAN for few-shot font style transfer, 2017](https://github.com/azadis/MC-GAN)
2. [Tet-gan: text effects transfer via stylization and destylization, AAAI 2019](https://github.com/williamyang1991/TET-GAN)

# Face detection & recognition
1. [MTCNN](https://github.com/timesler/facenet-pytorch/blob/master/examples/face_tracking.ipynb)

# Object detection & segmentation
- [SSD tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)
        - [A good list of SSD-like CNN-based segmentation (keras, but useful list)](https://github.com/mvoelk/ssd_detectors)
- [Detectron2 by FAIR](https://github.com/facebookresearch/detectron2)
- [UNet pytorch](https://github.com/milesial/Pytorch-UNet)
- [Segmentation Models in Pytorch](https://github.com/qubvel/segmentation_models.pytorch)
    - [Some competitions won with the library](https://github.com/qubvel/segmentation_models.pytorch/blob/master/HALLOFFAME.md)
    1. UNet, UNet++
    2. MANet
    3. LinkNet
    4. FPN
    5. PSPNet
    6. PAN
    7. DeepLab-v3 & DeepLab-v3+ 
    - [Cloths segmentation](https://github.com/ternaus/cloths_segmentation)
    - [Kaggle Competition: iMaterialist (Fashion) 2019 at FGVC6, Fine-Grained Visual Categorization 6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6

# Super-Pixel Segmentation
1. [SpixelFCN: Superpixel Segmentation with Fully Convolutional Network, github](https://github.com/fuy34/superpixel_fcn)
    - [CVPR'20 paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Superpixel_Segmentation_With_Fully_Convolutional_Networks_CVPR_2020_paper.pdf)
2. [Superpixels: An evaluation of the state-of-the-art, CVIU 2018, github](https://github.com/davidstutz/superpixel-benchmark)
 
### Superpixel (SLIC like)
1. [SLIC Superpixels](https://ivrlwww.epfl.ch/supplementary_material/RK_SLICSuperpixels/index.html)
    - source code is in opencv & skimage
3. [opencv slic](https://docs.opencv.org/3.4/d3/da9/classcv_1_1ximgproc_1_1SuperpixelSLIC.html#details)
    -  SLIC (Simple Linear Iterative Clustering) clusters pixels using pixel channels and image plane space to efficiently generate compact, nearly uniform superpixels. The simplicity of approach makes it extremely easy to use a lone parameter specifies the number of superpixels and the efficiency of the algorithm makes it very practical. Several optimizations are available for SLIC class: SLICO stands for "Zero parameter SLIC" and it is an optimization of baseline SLIC described in [1]. MSLIC stands for "Manifold SLIC" and it is an optimization of baseline SLIC described in [134].
    -  134: Intrinsic Manifold SLIC: A Simple and Efficient Method for Computing Content-Sensitive Superpixels, Yong-Jin Liu, Minging Yu, bing-Jun Li, Ying He, IEEE TPAMI
4. [Opencv SLIC tutorial with python](https://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/)
5. [Fast Multilevel Superpixel Segmentation (Fast-MSS), github](https://github.com/JordanMakesMaps/Fast-Multilevel-Superpixel-Segmentation)
6. [Power-SLIC: Diagram-based superpixel generation, 2020, paper only](https://arxiv.org/pdf/2012.11772.pdf)

# Neural Style Transfer
- [Neural Transfer using Pytorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

# Generative Adversarial Network
- [wikipedia](https://en.wikipedia.org/wiki/Generative_adversarial_network)
- [Pytorch-GAN, upto 2019](https://github.com/eriklindernoren/PyTorch-GAN)
- [StyleGan with limited data](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [Few-shot compositional font generation with dual memory, ECCV'20](https://github.com/clovaai/dmfont)

# Super-resolution
- [wikipedia](https://en.wikipedia.org/wiki/Super-resolution_imaging)

# Reinforcement Learning
- [Playing Starcraft II](http://bennycheung.github.io/adventures-in-deep-reinforcement-learning)
- [RL in computer games AI by Tomasz Mackoviak (youtube)](https://youtu.be/Y3gT3z2uVB8)
- [RL talk by Peeter Abbel](https://youtu.be/IXuHxkpO5E8)
- [RL game playing bots compete (youtube)](https://youtu.be/1-f51I231G0)
- 
# Self-Driving Cars

# Visual Question Answering (VQA)
- [Visual Question Answering Challenge 2021](https://visualqa.org/challenge.html)

# Sign-Language Recognition
- [2021 Workshop and Challenge](https://chalearnlap.cvc.uab.cat/workshop/42/program/)
- [Hongdong Li's talk](https://data.chalearnlap.cvc.uab.cat/AuTSL/webpage/presentations/2.Hongdong_Li.mp4)

# Recommendation System
- [Food Discovery with Uber Eats](https://eng.uber.com/uber-eats-graph-learning/#:~:text=The%20Uber%20Eats%20recommendation%20system,restaurants%2C%20in%20a%20scalable%20fashion.)
     - [How Uber uses Graph Neural Networks to recommend you food (youtube)](https://youtu.be/9O9osybNvyY)

# Chatbot
- [Chatbot Tutorial by PyTorch.org](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)


# Fine-Grained Visual Categorization
- [CVPR Workshop 2021](https://sites.google.com/view/fgvc8/papers)
      - [IndoFashion: Apparel Classification for Indian Ethnic Clothes](https://drive.google.com/file/d/112XZpH24gR2izr5bQmo6lJX80Z6OP_e6/view)
  
  
