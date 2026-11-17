# Multimodal Data Analysis (BAI-417) Lab

Lab work for **Multimodal Data Analysis (BAI-417)**, 7th semester, B.Tech ECE-AI (2023-2027), Indira Gandhi Delhi Technical University for Women (IGDTUW).

**Student:** Sneha Roychowdhury, ECE-AI 2
**Faculty:** Dr. Parikha Chawla

This repository holds the notebooks, code, datasets and the final lab file for all ten experiments. It is kept for documentation.

## About the subject

Real-world data rarely comes in one form. A product listing has text and photos, a lecture has video, audio and slides, and a social media post mixes a caption with an image. Multimodal Data Analysis is about working with these different data types, first one at a time and then together.

The lab follows that order:

1. **Preprocessing each modality** (Exp 1 to 4): turning raw text, images, audio and video into clean numerical features that a model can use, e.g. TF-IDF vectors, normalised and augmented images, MFCCs, sampled frames and CNN features.
2. **Combining modalities** (Exp 5, 6, 8): fusing features from two sources (text and image, text and audio, frames and subtitles) and checking whether the combination beats each source on its own.
3. **Search and retrieval** (Exp 7, 9, 10): extracting structured information from text, ranking documents for a query, and retrieving images with text queries (and the reverse) through a shared embedding space.

## Experiments

| No. | Experiment | Date | Notebook | Dataset |
|---|---|---|---|---|
| 1 | Text Data Preprocessing | 19/08/26 | [Exp01](exp01/Exp01_Text_Data_Preprocessing.ipynb) | [Own reviews](exp01) |
| 2 | Image Data Preprocessing | 02/09/26 | [Exp02](exp02/Exp02_Image_Data_Preprocessing.ipynb) | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) |
| 3 | Audio Data Preprocessing | 09/09/26 | [Exp03](exp03/Exp03_Audio_Data_Preprocessing.ipynb) | [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) |
| 4 | Video Data Preprocessing | 16/09/26 | [Exp04](exp04/Exp04_Video_Data_Preprocessing.ipynb) | [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) |
| 5 | Multimodal Data Integration | 23/09/26 | [Exp05](exp05/Exp05_Multimodal_Data_Integration.ipynb) | [Generated](exp05) |
| 6 | Emotion Recognition from Text and Audio | 01/10/26 | [Exp06](exp06/Exp06_Emotion_Recognition_Text_Audio.ipynb) | [Generated](exp06) |
| 7 | Named Entity Recognition | 19/10/26 | [Exp07](exp07/Exp07_Named_Entity_Recognition.ipynb) | [Generated](exp07) |
| 8 | Video Summarization with Text and Visuals | 28/10/26 | [Exp08](exp08/Exp08_Video_Summarization.ipynb) | [Generated](exp08) |
| 9 | Document Ranking | 12/11/26 | [Exp09](exp09/Exp09_Document_Ranking_TFIDF_BM25.ipynb) | [Generated](exp09) |
| 10 | Cross Modal Retrieval | 12/11/26 | [Exp10](exp10/Exp10_Cross_Modal_Retrieval.ipynb) | [Generated](exp10) |

### What each experiment does

1. **Text Data Preprocessing.** Cleans, tokenises, removes stopwords from and lemmatises customer reviews with spaCy, then builds Bag-of-Words and TF-IDF vectors. With one review every word gets the same weight; with five, TF-IDF starts picking out the words specific to each review.
2. **Image Data Preprocessing.** Resizes a CIFAR-10 image to 224 x 224, applies augmentation (flip, rotation, zoom, translation) and VGG16 normalisation, then extracts a 7 x 7 x 512 feature map with a pretrained VGG16.
3. **Audio Data Preprocessing.** Loads UrbanSound8K clips, reduces noise with spectral gating, resamples to 22,050 Hz and extracts 20 MFCCs per frame. The same pipeline is run on 250 clips to build a feature table.
4. **Video Data Preprocessing.** Samples 16 frames from UCF101 clips, resizes, crops and normalises them, and extracts optical flow, ResNet-50 frame features and R3D-18 clip features.
5. **Multimodal Data Integration.** Classifies sentiment from TF-IDF caption features and ResNet-18 image features, comparing text only, image only, early fusion and late fusion.
6. **Emotion Recognition from Text and Audio.** Combines TF-IDF text features with pitch, energy, spectral and MFCC audio features to classify four emotions.
7. **Named Entity Recognition.** Extracts people, organisations, places and dates from a news corpus with spaCy, then ranks the documents for search queries with TF-IDF and BM25.
8. **Video Summarization with Text and Visuals.** Detects shots from colour-histogram changes, picks keyframes, scores each shot's subtitles with TF-IDF and fuses both into a storyboard, text summary and shorter video.
9. **Document Ranking.** Implements TF-IDF and BM25 from scratch over an inverted index and evaluates them with Precision@5, Recall@5, MAP, MRR and nDCG@10.
10. **Cross Modal Retrieval.** Learns a shared text-image space with CCA and a contrastive two-tower model (the idea behind CLIP, written in NumPy) to retrieve images from captions and captions from images.

## Repository structure

```
exp01/ ... exp10/   one folder per experiment: notebook, dataset (where it is not public) and a README
lab-file/           the final lab file (PDF, 40 pages)
```

Experiments 2, 3 and 4 use public datasets, so their folders link to the official source instead of storing a copy. The datasets for experiments 5 to 10 are generated by the notebooks with fixed random seeds, so re-running a notebook gives the same data that is stored here.

## Running the notebooks

Every notebook was written for and run on **Google Colab**:

1. Open the notebook on GitHub and choose **Open in Colab**, or upload it to Colab.
2. Click **Runtime → Run all**. Each notebook installs anything Colab is missing and creates or downloads its own data, so no setup is needed.

Experiments 2, 4 and 5 download pretrained weights (VGG16, ResNet-50 and R3D-18, ResNet-18) on first run, and Experiment 3 downloads UrbanSound8K (about 6 GB), so they need an internet connection and take a few minutes longer. A GPU runtime helps for Experiments 2 to 4 but is not required.

Main libraries: pandas, NumPy, scikit-learn, spaCy, NLTK, TensorFlow/Keras, PyTorch/torchvision, librosa, noisereduce, OpenCV, scikit-image and Matplotlib.

## Lab file

The complete write-up of all ten experiments (objective, theory, algorithm, dataset, code, sample input and output, observations and conclusion) is in [lab-file/MDA_Lab_File_Sneha_Roychowdhury.pdf](lab-file/MDA_Lab_File_Sneha_Roychowdhury.pdf).
