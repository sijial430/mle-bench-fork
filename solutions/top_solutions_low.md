# Top Human Solutions for MLE-Bench Competitions (low.txt)

---

## 1. Aerial Cactus Identification

- **Kaggle slug:** `aerial-cactus-identification`
- **URL:** https://www.kaggle.com/competitions/aerial-cactus-identification
- **Dates:** ~Mar--Jul 2019 (playground competition)
- **Task:** Binary image classification -- determine whether a 32x32 pixel aerial thumbnail image contains a columnar cactus (*Neobuxbaumia tetetzo*).
- **Metric:** Area Under the ROC Curve (AUC-ROC)
- **Teams:** ~1,228
- **Data:** 17,500 training images (32x32 JPEG, 75% positive), 4,000 test images. Part of Mexico's VIGIA project for autonomous surveillance of protected natural areas.

### Top Solutions

This was a relatively easy playground competition -- many teams achieved near-perfect scores (0.9999--1.0).

**Perfect Score with fastai (ResNet-34 Transfer Learning):**
- **Model:** Pre-trained ResNet-34 via fastai's `cnn_learner`
- **Training:** `fit_one_cycle()` (1-cycle learning rate policy)
- **Augmentation:** Standard fastai transforms (flipping, rotation, lighting)
- **Key Insight:** Pretrained ResNet-34 with minimal tuning achieves perfect score

**Simple CNN (0.9999 AUC):**
- **Model:** Custom 5-layer CNN (32->64->128->256->512 channels), BatchNorm, Leaky ReLU, MaxPool
- **Training:** 25 epochs, batch size 128, LR 0.002, Adamax optimizer, CrossEntropyLoss
- **Validation accuracy:** 99.26%

**VGG-like CNN (0.9997 AUC):**
- **Model:** Custom VGG-style CNN, 3 pairs of Conv2D layers (32/64/128 filters)
- **Augmentation:** Rotation up to 60 degrees, zoom 0.2, horizontal/vertical flip

**Code:**
- [Cactus Finder: Perfect Score with fastai.vision](https://www.kaggle.com/code/abyaadrafid/cactus-finder-perfect-score-with-fastai-vision)
- [ROC AUC 0.9999 Simple CNN](https://www.kaggle.com/code/werooring/roc-auc-0-9999-simple-cnn-model)
- [PotatoSpudowski/CactiNet](https://github.com/PotatoSpudowski/CactiNet) -- EfficientNet-inspired custom model
- [sayakpaul/Aerial-Cactus-Identification](https://github.com/sayakpaul/Aerial-Cactus-Identification)

---

## 2. APTOS 2019 Blindness Detection

- **Kaggle slug:** `aptos2019-blindness-detection`
- **URL:** https://www.kaggle.com/competitions/aptos2019-blindness-detection
- **Dates:** ~Jun--Sep 2019
- **Organizer:** Asia Pacific Tele-Ophthalmology Society (APTOS) + Aravind Eye Hospital
- **Task:** Grade severity of diabetic retinopathy from retinal fundus photographs (5 classes: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative).
- **Metric:** Quadratic Weighted Kappa (QWK)
- **Teams:** ~2,943
- **Data:** 3,662 training images + external 2015 DR dataset (~35K images)

### Key Universal Techniques Across Top Solutions

- **Regression over Classification:** Nearly all top solutions framed as regression (MSE loss), predictions rounded/optimized for ordinal boundaries.
- **Pretraining on 2015 DR Data:** Critical for performance.
- **Pseudo-labeling:** First-stage model predictions on test set used as labels for retraining.
- **Models:** EfficientNet (B3-B5) and SE-ResNeXt (50, 101) dominated.

### 1st Place

- **Models:** Inception and SE-ResNeXt only
- **Preprocessing:** Minimal -- images resized to 512x512
- No detailed public writeup available.

### 11th Place -- 4uiiurz1

- **Models:** Two-level ensemble -- SE-ResNeXt50, SE-ResNeXt101, SENet154
- **Preprocessing:** Ben's crop (scale radius), 256x256 images
- **Training:** 30 epochs MSE loss + SGD + CosineAnnealingLR, then 10 epochs with RAdam + pseudo-labels
- **Scores:** Public 0.826, Private 0.930

### 37th Place -- tahsin314

- **Models:** Ensemble of 5 EfficientNets (B0-B5)
- **Preprocessing:** Circle cropping only
- **Score:** Private 0.926

**Code:**
- [4uiiurz1/kaggle-aptos2019](https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection) -- 11th place
- [tahsin314/40th_place_solution](https://github.com/tahsin314/40th_place_solution_aptos2019-blindness-detection)
- [DrHB/APTOS-2019-GOLD-MEDAL-SOLUTION](https://github.com/DrHB/APTOS-2019-GOLD-MEDAL-SOLUTION)
- [MamatShamshiev/Kaggle-APTOS-2019](https://github.com/MamatShamshiev/Kaggle-APTOS-2019-Blindness-Detection) -- 76th place, [blog](https://diyago.github.io/2019/10/04/kaggle-blindness.html)
- [arXiv:2003.02261](https://arxiv.org/abs/2003.02261) -- 54th place, QWK 0.925

---

## 3. Denoising Dirty Documents

- **Kaggle slug:** `denoising-dirty-documents`
- **URL:** https://www.kaggle.com/c/denoising-dirty-documents
- **Dates:** ~Jun--Oct 2015 (playground)
- **Task:** Remove noise (coffee stains, creases, watermarks) from scanned document images to recover clean text.
- **Metric:** RMSE on pixel intensity values
- **Data:** 144 training image pairs (dirty + clean), derived from UCI's NoisyOffice dataset. Only 8 different backgrounds, 2 coffee stains, 2 folded pages, etc. -- exploitable as information leakage.

### Top Solutions

**6th Place -- toshi-k (CNN):**
- **Model:** 6-layer CNN (96 channels, alternating 3x3 and 1x1 convolutions), Leaky ReLU
- **Framework:** Torch/Lua
- **Code:** [toshi-k/kaggle-denoising-dirty-documents](https://github.com/toshi-k/kaggle-denoising-dirty-documents)

**Colin Priest (11-part blog series):**
- Progressive development from linear regression to information leakage exploitation to DNNs
- **Key insight:** Exploiting the 8 shared backgrounds to estimate per-pixel background brightness, then rescaling contrast
- **Final model:** Ensemble of 3 deep learning models via **h2o** (R), outperformed CNNs
- **Blog:** [Part 1](https://colinpriest.com/2015/08/01/denoising-dirty-documents-part-1/) through [Part 11](https://colinpriest.com/2015/11/08/denoising-dirty-documents-part-11/)

**kapsdeep (Multi-approach):**
- Best: single-layer 3x3 convolution + RandomForestRegressor, RMSE 0.02656
- **Code:** [kapsdeep/Kaggle-Denoise-Dirty-Documents](https://github.com/kapsdeep/Kaggle-Denoise-Dirty-Documents)

---

## 4. Detecting Insults in Social Commentary

- **Kaggle slug:** `detecting-insults-in-social-commentary`
- **URL:** https://www.kaggle.com/c/detecting-insults-in-social-commentary
- **Dates:** Aug 7 -- Sep 17, 2012
- **Organizer:** Impermium
- **Task:** Binary classification -- predict whether a comment is insulting to a participant.
- **Metric:** AUC-ROC
- **Teams:** 154 participants, 1,235 entries
- **Prize:** $10,000 ($7,000 for 1st)

### 1st Place -- Vivek Sharma

- **Model:** SVC + Random Forest ensemble
- **Key Features:** Bad words list (from urbanoalvarez.es), phrase patterns ("you are/you're a/an [insult]"), text normalizations
- **Key Insight:** SVC outperformed regularized LR; structural phrase patterns were a key differentiator

### 2nd Place -- tuzzeg

- **Approach:** 110 base classifiers (Logistic Regression) combined via Random Forest stacking
- **Features:** Stem subsequences (66% importance), stem unigrams/bigrams (18%), character n-grams (11%), syntax features via Stanford Parser
- **Innovation:** Sentence-level models with max-across-sentences scoring improved AUC from 0.72 to 0.77
- **Code:** [tuzzeg/detect_insults](https://github.com/tuzzeg/detect_insults)

### 3rd Place -- Andrei Olariu

- **Approach:** 4-component ensemble: SVM on word n-grams, SVM on char n-grams, dictionary-based classifier, neural network meta-learner
- **Blog:** [My first Kaggle competition](http://webmining.olariu.org/my-first-kaggle-competition-and-how-i-ranked/)

**Code:**
- [amueller/kaggle_insults](https://github.com/amueller/kaggle_insults) -- 6th place, LinearSVC
- [cbrew/Insults](https://github.com/cbrew/Insults) -- 4th place, SGD classifier

---

## 5. Dog Breed Identification

*(See `reports/top_solutions_mix.md` for detailed coverage of this competition.)*

---

## 6. Dogs vs. Cats Redux: Kernels Edition

- **Kaggle slug:** `dogs-vs-cats-redux-kernels-edition`
- **URL:** https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition
- **Dates:** Sep 2, 2016 -- Mar 2, 2017 (~6 months, playground)
- **Task:** Binary image classification -- distinguish dogs from cats.
- **Metric:** Log Loss
- **Teams:** 1,314
- **Data:** 25,000 training images (12,500 each), 12,500 test images

### 3rd Place -- Marco Lugo

- **Models:** Bottleneck features from InceptionV3, ResNet50, VGG16, Xception fed to LightGBM and 5-layer NN; also 2 VGG-inspired CNNs from scratch
- **Key Finding:** Leaky ReLU and Randomized Leaky ReLU noticeably improved performance
- **Writeup:** [Kaggle Blog Interview](https://medium.com/kaggle-blog/dogs-vs-cats-redux-playground-competition-3rd-place-interview-marco-lugo-74893739b10f)

### 4th Place -- Bojan Tunguz

- **Models:** VGG16, VGG19, ResNet50/101/152/200/269, Inception V3, Xception (Keras + Facebook Torch ResNets)
- **Score Progression:** Simple CNN ~0.2x -> retrained features ~0.06x -> stacking ~0.05x -> fine-tuned 0.042 -> ensemble ~0.035
- **Writeup:** [Kaggle Blog Interview](https://medium.com/kaggle-blog/dogs-vs-cats-redux-playground-competition-winners-interview-bojan-tunguz-7233c12e03bf)

**Code:**
- [RomanKornev/dogs-vs-cats-redux](https://github.com/RomanKornev/dogs-vs-cats-redux) -- 15th, 99.7% accuracy
- [prajwalkr/dogsVScats](https://github.com/prajwalkr/dogsVScats) -- 60th, ResNet-50 + InceptionV3
- [shaoanlu/dogs-vs-cats-redux](https://github.com/shaoanlu/dogs-vs-cats-redux) -- Top 1.3%

---

## 7. Histopathologic Cancer Detection

- **Kaggle slug:** `histopathologic-cancer-detection`
- **URL:** https://www.kaggle.com/competitions/histopathologic-cancer-detection
- **Dates:** Nov 2018 -- Mar 30, 2019 (playground)
- **Task:** Binary classification -- identify metastatic cancer in 96x96 pixel patches from lymph node sections (PatchCamelyon / PCam dataset).
- **Metric:** AUC-ROC
- **Teams:** ~1,100+
- **Data:** ~220,000 training images, ~57,000 test images, 50/50 balanced. Label based on center 32x32 region.

### Top Solutions

**15th Place -- Sergey Kolchenko & Alex Donchuk:**
- **Model:** SE-ResNeXt50, 10-fold CV, SGD + ReduceOnPlateau
- **Critical Insight:** Data leakage from same whole-slide images appearing in train/validation; WSI-based group splits fixed this
- **Blog:** [Medium](https://sergeykolchenko.medium.com/histopathologic-cancer-detection-as-image-classification-using-pytorch-557aab058449)

**Top 1% -- Ivan Panshin:**
- **Techniques:** Stochastic Weight Averaging (SWA), TTA-4 via ttach library, multi-model ensemble
- **Score:** >0.98 AUC-ROC on private test
- **Code:** [ivanpanshin/hist_cancer](https://github.com/ivanpanshin/hist_cancer)

**ResNet50 + fastai (98.6% accuracy):**
- **Model:** ResNet50 via fast.ai, 1cycle policy, discriminative learning rates
- **Blog:** [humanunsupervised.com](https://humanunsupervised.github.io/humanunsupervised.com/pcam/pcam-cancer-detection.html)

---

## 8. Jigsaw Toxic Comment Classification Challenge

- **Kaggle slug:** `jigsaw-toxic-comment-classification-challenge`
- **URL:** https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
- **Dates:** ~Late 2017 -- Mar 2018
- **Organizer:** Jigsaw / Conversation AI (Alphabet/Google)
- **Task:** Multi-label text classification -- predict 6 toxicity categories (toxic, severe_toxic, obscene, threat, insult, identity_hate) for Wikipedia comments.
- **Metric:** Mean column-wise ROC AUC
- **Teams:** ~4,551
- **Prize:** $35,000
- **Data:** 159,571 training comments, 153,164 test comments

### 1st Place -- "Toxic Crusaders" (Chun Ming Lee)

- **Models:** BiGRU, CapsuleNet, DeepMoji, HAN, DPCNN, VDCNN, character-level CNNs
- **Embeddings:** FastText crawl-300d-2M, GloVe 840B, ConceptNet NumberBatch
- **Data Augmentation:** Back-translation (French, German, Spanish) -- critical for minority labels
- **Ensemble:** Multi-level stacking: 10-fold OOF predictions -> LightGBM stacking
- **Key Insight:** FastText's subword OOV handling added ~0.002 AUC over zero-vector replacement
- **Writeup:** [1st Place Solution (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557)

### 2nd Place -- neongen

- **Models:** Ensemble of RNN + CNN + LightGBM + XGBoost
- **Embeddings:** BPEmb (Byte-Pair Embeddings) with 200K merges performed best
- **Key Insight:** Minimal preprocessing sometimes better; GBM included for diversity despite lower individual performance

### 3rd Place -- Bojan Tunguz

- **Techniques:** Pseudo-labeling, multi-model ensembling, NBSVM + neural networks

**Code:**
- [zake7749/DeepToxic](https://github.com/zake7749/DeepToxic) -- 27th place, top 1%, Pooled RNN + Kmax CNN
- [soham97/Toxic-Comment-Classification](https://github.com/soham97/Toxic-Comment-Classification-Challenge-NLP) -- 15th place
- [minerva-ml/open-solution-toxic-comments](https://github.com/minerva-ml/open-solution-toxic-comments)
- [unitaryai/detoxify](https://github.com/unitaryai/detoxify) -- Trained models for all Jigsaw challenges

---

## 9. Leaf Classification

- **Kaggle slug:** `leaf-classification`
- **URL:** https://www.kaggle.com/competitions/leaf-classification
- **Dates:** ~Aug 2016 -- Feb 2017 (playground)
- **Task:** Multi-class classification of 99 leaf species from pre-extracted features (margin, shape, texture).
- **Metric:** Multi-class log loss
- **Teams:** ~1,500
- **Data:** 990 training samples (10 per species), 594 test samples. Features already extracted (no raw images needed for baseline).

### Top Solutions

- **Dominant Approach:** Logistic Regression achieved near-perfect results due to pre-extracted features
- **Best single model:** Logistic Regression with `solver='lbfgs'`, `multi_class='multinomial'`, `C=500-1000`
- **Ensemble:** Logistic Regression + Random Forest + KNN + SVC
- **Neural Network approach:** Simple NN (2 hidden layers with dropout) also competitive
- **Key Insight:** Pre-extracted features made this a feature-based classification problem, not a vision problem. Simple models dominated.

**Code:**
- [SamDuan/Leaf-Classification-Kaggle](https://github.com/SamDuan/Leaf-Classification-Kaggle) -- Multiple approaches
- [lorinanthony/KAGGLE-leaf-classification](https://github.com/lorinanthony/KAGGLE-leaf-classification) -- Logistic Regression
- [rafaelzingalo/leaf_classification_kaggle](https://github.com/rafaelzingalo/leaf_classification_kaggle) -- R xgboost

---

## 10. MLSP 2013 Bird Classification

- **Kaggle slug:** `mlsp-2013-birds`
- **URL:** https://www.kaggle.com/competitions/mlsp-2013-birds
- **Dates:** ~2013 (associated with MLSP 2013 workshop)
- **Task:** Multi-label audio classification -- identify bird species present in 10-second audio recordings.
- **Metric:** Micro-averaged AUC (row-averaged AUC across species)
- **Teams:** ~79
- **Data:** Audio recordings from field conditions, spectrogram-based features provided

### Top Solutions

**1st Place -- Gabor Fodor (beluga):**
- **Approach:** Contrast-enhanced spectrogram templates + template matching with cross-correlation
- **Classifier:** Gradient boosting on matched template features
- **Key Innovation:** Creating species-specific audio templates and using cross-correlation similarity as features
- **AUC:** ~0.95

**2nd Place -- Sander Dieleman:**
- **Approach:** Spherical k-means feature learning on spectrograms
- **Classifiers:** SVM and Random Forest on learned features
- **Blog:** [Classifying bird sounds with deep learning](https://benanne.github.io/2014/08/05/spotify-cnns.html) (referenced in later work)

**General techniques:** Binary Relevance + Random Forest, spectrogram feature extraction, template matching

---

## 11. New York City Taxi Fare Prediction

- **Kaggle slug:** `new-york-city-taxi-fare-prediction`
- **URL:** https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction
- **Dates:** ~2018 (playground)
- **Task:** Regression -- predict taxi fare amount from pickup/dropoff coordinates, datetime, and passenger count.
- **Metric:** RMSE
- **Teams:** ~1,463
- **Data:** ~55 million training rows, 9,914 test rows

### Top Solutions

- **Dominant Model:** LightGBM (and XGBoost)
- **Critical Feature Engineering:**
  - Haversine distance between pickup and dropoff
  - Airport proximity features (JFK, LaGuardia, Newark)
  - Geographic clustering (HDBScan on coordinates)
  - Temporal features (hour, day of week, month, year)
  - Manhattan distance approximation
  - Distance to known landmarks/hotspots
- **Key Insight:** Feature engineering on geographic coordinates was the main differentiator. Tree-based models with well-engineered features dominated.

**Code:**
- Multiple high-scoring notebooks on Kaggle using LightGBM with geographic feature engineering

---

## 12. NOMAD 2018 -- Predict Transparent Conductors

- **Kaggle slug:** `nomad2018-predict-transparent-conductors`
- **URL:** https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors
- **Dates:** ~2018
- **Task:** Predict formation energy and bandgap of transparent conductor materials from crystal structures.
- **Metric:** RMSLE (Root Mean Squared Logarithmic Error)
- **Teams:** Moderate-sized competition
- **Data:** Crystal structures (Al, Ga, In oxides in sesquioxide, spinel, and other configurations)

### 1st Place

- **Method:** Crystal graph n-gram descriptors + Kernel Ridge Regression (KRR)
- **Key Innovation:** Novel crystal structure featurization using graph-based n-grams
- **Published in:** Nature npj Computational Materials

### 2nd Place

- **Method:** Compositional descriptors + LightGBM
- **Approach:** Hand-crafted material composition features with gradient boosting

### 3rd Place

- **Method:** SOAP (Smooth Overlap of Atomic Positions) descriptors + Neural Network
- **Approach:** Physics-based atomic environment descriptors fed into NN

---

## 13. Plant Pathology 2020 (FGVC7)

- **Kaggle slug:** `plant-pathology-2020-fgvc7`
- **URL:** https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7
- **Dates:** 2020 (part of FGVC7 workshop at CVPR 2020)
- **Task:** Classify apple leaf images into 4 classes: healthy, scab, rust, multiple diseases.
- **Metric:** Mean column-wise AUC-ROC
- **Teams:** 1,317
- **Data:** Apple leaf images from real orchards

### 1st Place -- Alipay Team

- **Model:** Knowledge distillation with se_resnext50
- **Score:** Private AUC 0.98445
- **Key Technique:** Teacher-student knowledge distillation framework

### General Top Techniques

- **Models:** SE-ResNeXt50, EfficientNet variants
- **Augmentation:** Heavy augmentation (CutMix, Mixup, color jittering, geometric transforms)
- **Training:** 5-fold cross-validation, progressive resizing, cosine annealing

**Code:**
- Various Kaggle notebooks using EfficientNet and SE-ResNeXt with heavy augmentation

---

## 14. Random Acts of Pizza

- **Kaggle slug:** `random-acts-of-pizza`
- **URL:** https://www.kaggle.com/competitions/random-acts-of-pizza
- **Dates:** ~2014 (knowledge competition)
- **Task:** Binary classification -- predict whether a Reddit post requesting free pizza (from r/Random_Acts_Of_Pizza) will receive pizza.
- **Metric:** AUC-ROC
- **Teams:** 463
- **Data:** Reddit posts with text, metadata (account age, karma, etc.)

### Top Solutions

- **Models:** XGBoost / Gradient Boosted Trees dominated
- **Features:** Text features (sentiment, length, politeness indicators) + metadata (account age, karma, subreddit activity)
- **Data Leakage:** Significant leakage issue -- some features allowed identification of specific users/posts, inflating scores. Best legitimate AUC ~0.84-0.89.
- **Key Insight:** This was more of a feature engineering challenge than a modeling challenge.

---

## 15. RANZCR CLiP -- Catheter and Line Position Challenge

- **Kaggle slug:** `ranzcr-clip-catheter-line-classification`
- **URL:** https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification
- **Dates:** ~2021
- **Organizer:** Royal Australian and New Zealand College of Radiologists (RANZCR)
- **Task:** Classify the presence and position of 11 catheter/line types in chest X-rays (11 binary labels).
- **Metric:** Mean AUC across 11 labels
- **Teams:** ~1,549
- **Prize:** $50,000

### 1st Place

- **Approach:** Two-step pipeline:
  1. **Segmentation (UNet++):** Segment catheter/line locations in X-rays
  2. **Classification (EfficientNet):** Classify presence and position using segmentation maps as auxiliary input
- **Key Innovation:** Using segmentation as an intermediate representation significantly improved classification accuracy
- **Models:** UNet++ for segmentation, EfficientNet (B5-B7) for classification

### General Top Techniques

- **External data:** CheXpert and NIH Chest X-ray datasets for pretraining
- **Augmentation:** Heavy medical imaging augmentation (rotation, brightness, elastic deformation)
- **Ensemble:** Multi-model ensemble with TTA

---

## 16. SIIM-ISIC Melanoma Classification

- **Kaggle slug:** `siim-isic-melanoma-classification`
- **URL:** https://www.kaggle.com/competitions/siim-isic-melanoma-classification
- **Dates:** ~2020
- **Organizer:** SIIM + ISIC (International Skin Imaging Collaboration)
- **Task:** Binary classification -- identify melanoma in dermoscopic images.
- **Metric:** AUC-ROC
- **Teams:** 3,314
- **Prize:** $30,000
- **Data:** Dermoscopic images + patient metadata (age, sex, anatomical site)

### 1st Place

- **Models:** Ensemble of 18 models: EfficientNet variants (B3-B7), SE-ResNeXt, ResNeSt
- **Ensemble Method:** Rank-based averaging across all models
- **Training:**
  - Multi-class targets (framing melanoma detection alongside other diagnostic labels) improved binary melanoma prediction
  - External data from previous ISIC challenges
  - Heavy augmentation (CutOut, CutMix, advanced color augmentation)
- **Key Insight:** Training with multi-class auxiliary targets regularized the model and improved melanoma-specific AUC

### General Top Techniques

- **Models:** EfficientNet family dominated
- **Metadata:** Patient metadata (age, sex, site) concatenated with image features in final classification layer
- **External data:** Previous years' ISIC datasets
- **Augmentation:** Hair removal augmentation, microscope artifact removal

---

## 17. Spooky Author Identification

- **Kaggle slug:** `spooky-author-identification`
- **URL:** https://www.kaggle.com/competitions/spooky-author-identification
- **Dates:** ~Oct 2017 -- Jan 2018 (playground)
- **Task:** 3-class text classification -- identify author (Edgar Allan Poe, HP Lovecraft, Mary Shelley) from sentence excerpts.
- **Metric:** Multi-class log loss
- **Teams:** 1,245
- **Data:** Sentences from works of the three authors

### Top Solutions

- **Dominant Approach:** Character n-gram TF-IDF + Logistic Regression was surprisingly competitive
- **Best techniques:**
  - TF-IDF on character n-grams (2-6 grams) + logistic regression
  - Word-level TF-IDF + Naive Bayes / SVM
  - LSTM/GRU on word sequences
  - Ensemble of TF-IDF models + neural models
- **Key Insight:** Character-level n-grams captured author-specific writing style (word choices, punctuation patterns) very effectively. Simple TF-IDF + LR was hard to beat.

**Code:**
- Many Kaggle notebooks using TF-IDF + LogReg as competitive baseline

---

## 18. Tabular Playground Series -- December 2021

- **Kaggle slug:** `tabular-playground-series-dec-2021`
- **URL:** https://www.kaggle.com/competitions/tabular-playground-series-dec-2021
- **Dates:** Dec 2021 (monthly playground)
- **Task:** Multi-class classification -- forest cover type prediction (synthetic version of Covertype dataset).
- **Metric:** Accuracy
- **Data:** Synthetic data generated via CTGAN from the original Covertype dataset

### Top Solutions

- **Dominant Models:** GBDT ensembles (XGBoost, LightGBM, CatBoost)
- **Key Techniques:**
  - Feature engineering on elevation, soil types, wilderness areas
  - Pseudo-labeling with confident test predictions
  - Using original Covertype dataset as additional training data
  - Stacking multiple GBDT models
- **Key Insight:** The synthetic nature of the data (CTGAN-generated) introduced artifacts that could be exploited

---

## 19. Tabular Playground Series -- May 2022

- **Kaggle slug:** `tabular-playground-series-may-2022`
- **URL:** https://www.kaggle.com/competitions/tabular-playground-series-may-2022
- **Dates:** May 2022 (monthly playground)
- **Task:** Binary classification with engineered features.
- **Metric:** AUC-ROC
- **Teams:** ~1,151
- **Data:** Tabular features including a critical string feature `f_27`

### Top Solutions

- **Key Discovery:** Feature `f_27` was a string that could be decomposed into individual character features, dramatically improving performance
- **Feature Engineering:** Pairwise feature interactions between decomposed `f_27` characters and other features
- **Winning Model:** Neural Network + LightGBM ensemble
- **Key Insight:** Understanding the structure of `f_27` was the critical differentiator between top and average solutions

---

## 20. Text Normalization Challenge -- English Language

- **Kaggle slug:** `text-normalization-challenge-english-language`
- **URL:** https://www.kaggle.com/competitions/text-normalization-challenge-english-language
- **Dates:** ~Sep--Nov 2017
- **Organizer:** Google Speech & Language Algorithms Team (Richard Sproat & Kyle Gorman)
- **Task:** Convert written English text to spoken form (TTS normalization) -- normalize numbers, dates, abbreviations, etc.
- **Metric:** Sentence-level accuracy
- **Teams:** ~260
- **Data:** Text with semiotic class annotations and normalized outputs

### Top Solutions

- **Key Finding (Sproat & Gorman summary):** Top English solutions involved quite a bit of **manual grammar engineering** -- special rule sets for different semiotic classes (measures, dates, cardinals, ordinals, etc.), with supervised classifiers used to identify the appropriate semiotic class for tokens.
- **Some teams used neural approaches:** LSTM encoder/decoder, seq2seq models
- **Foundational paper:** Sproat & Jaitly (2016), "RNN Approaches to Text Normalization: A Challenge" ([arXiv:1611.00068](https://arxiv.org/abs/1611.00068))

**Code:**
- [georgercarder/Kaggle-Text-Normalization](https://github.com/georgercarder/Kaggle-Text-Normalization-Challenge--English-version-finalsubmission) -- 99.09% accuracy
- [caravanuden/text_norm](https://github.com/caravanuden/text_norm) -- LSTM encoder/decoder
- [shauryr/google_text_normalization](https://github.com/shauryr/google_text_normalization) -- RNN approach

**Blog:** [A Brief Summary of the Kaggle Text Normalization Challenge (Kaggle Blog)](https://medium.com/kaggle-blog/a-brief-summary-of-the-kaggle-text-normalization-challenge-11797b7e696f)

---

## 21. Text Normalization Challenge -- Russian Language

- **Kaggle slug:** `text-normalization-challenge-russian-language`
- **URL:** https://www.kaggle.com/competitions/text-normalization-challenge-russian-language
- **Dates:** ~Sep--Nov 2017
- **Organizer:** Google Speech & Language Algorithms Team
- **Task:** Convert written Russian text to spoken form (TTS normalization).
- **Metric:** Sentence-level accuracy
- **Teams:** ~162

### 1st Place -- University of Stuttgart

- **Method:** Sequence-to-sequence Neural Machine Translation (NMT)
- **Approach:** Treated text normalization as a translation problem (written form -> spoken form)
- **Key Innovation:** Applied NMT techniques to text normalization, outperforming rule-based systems

### 2nd Place

- **Method:** Fully convolutional network
- **Approach:** Used convolutional architecture instead of RNN for sequence transduction

### 6th Place

- **Method:** Differentiable Neural Computers (DNC)
- **Notable for:** Using a relatively exotic architecture (DNC) competitively

---

## 22. The ICML 2013 Whale Challenge -- Right Whale Redux

- **Kaggle slug:** `the-icml-2013-whale-challenge-right-whale-redux`
- **URL:** https://www.kaggle.com/competitions/the-icml-2013-whale-challenge-right-whale-redux
- **Dates:** ~2013 (associated with ICML 2013)
- **Task:** Binary audio classification -- detect right whale up-calls in underwater acoustic recordings.
- **Metric:** AUC-ROC
- **Teams:** ~26
- **Data:** 2-second audio clips, labeled for presence of right whale up-calls. Famous case study on data leakage.

### 1st Place -- Daniel Kridler

- **Method:** Contrast-enhanced spectrograms + template matching + gradient boosting
- **Approach:**
  1. Convert audio to spectrograms with contrast enhancement
  2. Create whale call templates via cross-correlation
  3. Use template matching features in a gradient boosted tree classifier
- **Key Innovation:** Spectrogram contrast enhancement and template-based feature extraction

### 2nd Place -- Sander Dieleman

- **Method:** Spherical k-means feature learning + SVM/Random Forest
- **Approach:** Unsupervised feature learning on spectrograms via spherical k-means, then supervised classification

### 3rd Place -- Florian Nouri

- **Method:** CNN on spectrograms
- **Approach:** One of the early applications of CNNs to audio classification via spectrogram images

### Data Leakage

This competition became a famous case study on data leakage -- certain patterns in the audio data allowed competitors to identify test examples without actually detecting whale calls.

---

## Notes

- For playground competitions (aerial-cactus, denoising, dogs-vs-cats, histopathologic-cancer, leaf-classification, spooky-author, tabular-playground), formal winner writeups are generally not published.
- For older competitions (detecting-insults 2012, mlsp-2013-birds, icml-2013-whale), documentation is sparse as these predate modern Kaggle solution-sharing culture.
- The most thoroughly documented competitions are APTOS 2019, Jigsaw Toxic Comment, RANZCR CLiP, and SIIM-ISIC Melanoma, all of which were featured competitions with prizes and required winner writeups.
