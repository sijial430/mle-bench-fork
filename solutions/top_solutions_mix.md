# Top Human Solutions for MLE-Bench Competitions (mix.txt)

---

## 1. Google Smartphone Decimeter Challenge 2022

- **Kaggle slug:** `smartphone-decimeter-2022`
- **URL:** https://www.kaggle.com/competitions/smartphone-decimeter-2022
- **Dates:** May 3, 2022 -- Jul 29-30, 2022 (~3 months)
- **Co-sponsor:** Institute of Navigation (ION)
- **Task:** Improve GNSS positioning accuracy on smartphones to decimeter-level.
- **Metric:** 50th/95th percentile horizontal positioning error (meters)
- **Data:** Raw GNSS measurements (pseudorange, Doppler, carrier phase / accumulated delta range) and IMU data from five smartphone types (Google Pixel 4, Pixel 5, Pixel 6 Pro, Samsung Galaxy S20 Ultra, Xiaomi Mi8), collected in the Los Angeles area.

### 1st Place -- Taro Suzuki (Chiba Institute of Technology, Japan)

**Paper:** "1st Place Winner of the Smartphone Decimeter Challenge: Two-Step Optimization of Velocity and Position Using Smartphone's Carrier Phase Observations"
**Presented at:** ION GNSS+ 2022, Denver, Colorado, September 2022.

**Key Techniques:**

- **Core Method: Two-Step Factor Graph Optimization (FGO).** The velocity and position optimization processes are separated into two distinct steps:
  1. **Step 1 -- Velocity Estimation:** Velocity is first estimated from GNSS Doppler observations via factor graph optimization. Doppler observations are also used to detect and exclude cycle slips in carrier-phase observations. A robust M-estimator (Huber function) is applied to reject multipath-corrupted Doppler measurements during optimization. Outliers in the estimated velocity are excluded and missing values are interpolated.
  2. **Step 2 -- Position Estimation:** The cleaned velocity estimates from Step 1 are used as loose constraints between state nodes in the position-optimization factor graph. Position is estimated using pseudorange observations, corrected pseudorange from GNSS reference (CORS) stations, and time-differenced carrier phase (TDCP) observations as additional constraints. This two-step separation makes position estimation more robust and accurate.
- **Observations Used:** Pseudorange, pseudorange rate (Doppler), and accumulated delta range (carrier phase / ADR) from multiple GNSS constellations.
- **Software Stack:**
  - **GTSAM** (Georgia Tech Smoothing and Mapping) -- C++ library used as the graph optimization backend.
  - **RTKLIB** -- used for general GNSS computations (satellite position calculation, atmospheric corrections, etc.).
- **Handling Challenging Environments:** The method handles GNSS-denied areas (tunnels, elevated structures) and severe multipath in urban environments. The robust error model (Huber M-estimator) automatically down-weights corrupted observations.
- **Result:** ~1.37 meters mean horizontal positioning error.

**Links:**
- ION publication: https://www.ion.org/publications/abstract.cfm?articleID=18377
- ResearchGate: https://www.researchgate.net/publication/364602532
- Full journal paper (MDPI Sensors, open access): https://www.mdpi.com/1424-8220/23/3/1205
- PubMed Central: https://pmc.ncbi.nlm.nih.gov/articles/PMC9919037/

**Code:**
- gtsam_gnss (open-source FGO package for GNSS): https://github.com/taroz/gtsam_gnss
- gsdc2023 (code for the 2023 challenge, same FGO approach): https://github.com/taroz/gsdc2023

### 2nd Place -- Dai S. (Kaggle Grandmaster)

**Paper:** "2nd Place Winner of the Smartphone Decimeter Challenge: Improving Smartphone GNSS Positioning Using Gradient Descent Method"
**Presented at:** ION GNSS+ 2022, Denver, Colorado, September 2022, pp. 2321-2328.

**Key Techniques:**

- **Core Method: Global Optimization via Gradient Descent.** Formulated the positioning problem as a global optimization over the entire driving track, solved iteratively using gradient descent.
- **Loss Function Design:** Custom loss functions encoding physical and geometrical constraints:
  - Pseudorange loss: constraining positions to be consistent with satellite pseudorange observations
  - Pseudorange rate (Doppler) loss: constraining velocity estimates
  - Accumulated carrier phase (ADR) loss: using carrier phase time-differences for high-precision relative positioning between epochs
  - Phone speed and acceleration constraints: physical constraints on vehicle dynamics to smooth the trajectory
- **Deep Learning Correction Model:** After gradient optimization, a deep learning model correlates positioning error with features such as orientation, velocity, acceleration, distance from reference stations, and phone model, producing a correction.
- **Background:** Top Kaggle grandmaster with no prior GNSS background; self-taught in ~50 days.

**Links:**
- ION publication: https://www.ion.org/publications/abstract.cfm?articleID=18380
- No public code repository identified.

### 3rd Place -- Tim Everett (rtklibexplorer)

**Paper:** "3rd Place Winner: 2022 Smartphone Decimeter Challenge: An RTKLIB Open-Source Based Solution"
**Presented at:** ION GNSS+ 2022, Denver, Colorado, September 2022.

**Key Techniques:**

- **Core Method: Post-Processed Kinematic (PPK) using a Modified RTKLIB.** Adapted the open-source RTKLIB software for lower-quality smartphone GNSS measurements.
- **Key RTKLIB Modifications:**
  - Cycle slip detection using Doppler measurements instead of carrier phase
  - Stochastic model tuning for noisier smartphone measurements
  - Configuration optimization for smartphone data characteristics
- **Base Station Data:** Nearby CORS stations for differential (PPK) processing.
- **Ensemble of Configurations:** Two different configuration files merged for improvement.
  - Single configuration: 1.593 / 1.743 (private/public leaderboard)
  - Ensemble: 1.608 / 1.715

**Links:**
- ION publication: https://www.ion.org/publications/abstract.cfm?articleID=18376
- Blog post: https://rtklibexplorer.wordpress.com/2022/06/06/google-smartphone-decimeter-challenge-2022/

**Code:**
- RTKLIB smartphone release: https://github.com/rtklibexplorer/RTKLIB/releases/tag/b34e_smartphone
- rtklib-py (Python PPK implementation): https://github.com/rtklibexplorer/rtklib-py

### Summary Comparison

| Aspect | 1st (Suzuki) | 2nd (Dai) | 3rd (Everett) |
|---|---|---|---|
| **Method** | Factor Graph Optimization (two-step) | Gradient Descent global optimization | Post-Processed Kinematic (PPK) |
| **Software** | GTSAM + RTKLIB | Custom framework | Modified RTKLIB |
| **Key Innovation** | Separate velocity/position optimization; Huber M-estimator | Physics-based loss functions + DL correction | Adapting RTKLIB for smartphone data |
| **Observations** | Pseudorange, Doppler, carrier phase | Pseudorange, Doppler, carrier phase + IMU | Pseudorange, Doppler, carrier phase |
| **Open Code** | gtsam_gnss (later release) | None | RTKLIB smartphone release + rtklib-py |
| **Background** | GNSS researcher | Kaggle grandmaster (ML/CV) | GNSS / RTKLIB expert |

---

## 2. Dog Breed Identification

- **Kaggle slug:** `dog-breed-identification`
- **URL:** https://www.kaggle.com/competitions/dog-breed-identification
- **Dates:** ~Oct/Nov 2017 -- Feb 27, 2018 (~3-4 months, playground competition)
- **Task:** Classify 120 dog breeds from images (ImageNet subset).
- **Metric:** Multi-class log loss.
- **Teams:** ~1,286

### Top Leaderboard

The top teams achieved **0.00000** log loss (perfect score). This was a playground competition where external data (Stanford Dogs Dataset) was allowed.

### Winning Approach Pattern

**1st-tier solutions (score ~0.00):**

- **Models:** Ensemble of multiple pretrained CNNs: InceptionV3, ResNet152_v1, DenseNet161, NASNet
- **Technique:** Global Average Pooling (GAP) transfer learning -- extract bottleneck features from multiple pretrained models, concatenate, train a small classifier on top
- **External data:** Stanford Dogs Dataset used for additional training
- **Key insight:** Using multiple model features concatenated dramatically boosts performance
- **Framework:** MXNet/Gluon or TensorFlow/Keras

**Typical pipeline:**
1. Extract features from 3+ pretrained models (InceptionV3, ResNet152, DenseNet161)
2. Concatenate features (~70MB preprocessed)
3. Train 2-layer FC classifier with dropout
4. Use Stanford Dogs Dataset as extra training data
5. Test-time augmentation

**Notable repos:**
- [ypwhs/DogBreed_gluon](https://github.com/ypwhs/DogBreed_gluon) -- Score 0.00398 (MXNet, InceptionV3 + ResNet152 + DenseNet161 + Stanford Dogs)
- [fenglf/Kaggle-dog-breed-classification](https://github.com/fenglf/Kaggle-dog-breed-classification) -- Score 0.00000 (GAP + pair training with multiple architectures)
- [GodsDusk/Dog-Breed-Identification](https://github.com/GodsDusk/Dog-Breed-Identification) -- 57th/1286 (top 5%), TF/Keras, ResNet + Inception

---

## 3. Facebook Recruiting III -- Keyword Extraction

- **Kaggle slug:** `facebook-recruiting-iii-keyword-extraction`
- **URL:** https://www.kaggle.com/competitions/facebook-recruiting-iii-keyword-extraction
- **Dates:** ~Aug 31, 2013 -- ~Dec 20, 2013 (~4 months)
- **Task:** Predict Stack Overflow tags from question title + body (multi-label classification).
- **Metric:** Mean F1 score.
- **Teams:** 366
- **Data:** ~6M training questions with Id, Title, Body, and Tags.

### Winning Approaches

No formal winner writeups were published for this older (2013) competition, but key approaches include:

**Top-performing techniques:**

- **TF-IDF + Classifier:** Encode posts with TF-IDF, use One-vs-Rest classifiers (Logistic Regression, Complement Naive Bayes) for multi-label prediction
- **Tag frequency heuristics:** Find tags popular in titles, use apriori algorithm for tag co-occurrence, check if popular tags appear in question body
- **Tag sampling:** Focus on top ~600 tags (2% of total tags) while covering 91% of questions, making training tractable
- **Feature engineering:** Title words, body words, code block presence, tag frequency statistics

**Notable code repos:**
- [alexeyza/Kaggle-Facebook3](https://github.com/alexeyza/Kaggle-Facebook3)
- [littleDing/kaggle/.../README.md](https://github.com/littleDing/kaggle/blob/master/facebook-recruiting-iii-keyword-extraction/README.md)
- [DhruvMakwana/Facebook-Recruiting-III-Keyword-Extraction](https://github.com/DhruvMakwana/Facebook-Recruiting-III-Keyword-Extraction)
- [alexf-a/fb-keyword-extraction](https://github.com/alexf-a/fb-keyword-extraction)

**Blog reference:** [Alex Minnaar's series](http://alexminnaar.com/facebook-recruiting-iii-keyword-extraction-part-4.html)

---

## 4. Billion Word Imputation

- **Kaggle slug:** `billion-word-imputation`
- **URL:** https://www.kaggle.com/competitions/billion-word-imputation
- **Dates:** May 8, 2014 -- May 1, 2015 (~12 months)
- **Task:** Find and insert a randomly removed word back into sentences from the Billion Word Corpus.
- **Metric:** Mean Levenshtein distance to original sentences.

### Top Solutions

**4th Place -- Cantab (Cambridge-based team)**

- **Method:** Combined a well-trained RNN language model with a KN 5-gram model
- **Process:** Language model generates all possible word insertions at all positions, returns top 100 most likely sentences with likelihoods
- **Decision rule:** If posterior probability of top candidate > 0.5, use that sentence; otherwise leave unchanged
- **Training time:** ~2 weeks
- **Discussion:** https://www.kaggle.com/c/billion-word-imputation/discussion/12977

**General winning techniques:**

- **N-gram models:** Interpolated Kneser-Ney 5-gram (100 CPUs, perplexity 243.2)
- **N-gram + POS tagging:** N-grams up to length 6 combined with Stanford POS tagger to determine missing word location
- **RNN-based:** Baseline RNN with hidden layer of size 100
- **Word2Vec + HMM/NLP Parser:** Word embeddings for candidate generation
- **Key insight:** "Guess conservatively if not confident about insertion position; guess short words more aggressively than long words"

**Code repos:**
- [timpalpant/KaggleBillionWordImputation](https://github.com/timpalpant/KaggleBillionWordImputation) -- N-gram approach (GPL-3.0)
- [johanlindberg/Billion-word-imputation](https://github.com/johanlindberg/Billion-word-imputation)
- [shawntan/billion-word-imputation](https://github.com/shawntan/billion-word-imputation)

**Academic references:**
- [IIT Kanpur report](https://cse.iitk.ac.in/users/cs365/2015/_submissions/mudgal/report.pdf)
- [Stanford CS224D report (RNN approach)](https://cs224d.stanford.edu/reports/ManiArathi.pdf)

---

## Notes

- For the older competitions (Facebook Recruiting III, Billion Word Imputation), formal winner writeups are scarce since these predate the modern Kaggle culture of detailed solution sharing.
- The Dog Breed Identification competition, being a playground competition, also lacks formal winner posts but has many high-quality open-source solutions.
- The Smartphone Decimeter Challenge 2022 has the most detailed documentation, with published ION papers and open-source code from all three top finishers.
