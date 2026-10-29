---
title: "Signer-Independent Evaluation and Russian-to-Uzbek Transfer for Isolated Sign Recognition with a Compact Part-Token Pose Encoder"
status: "Draft v1.0 (29 October 2026): full text; author data, ethics approval and repository link to be added."
---

::: authors
Mirsaid A. Yusupov ^1,\*^ and M. K. Onarkulov ^2^ [full English names and affiliations — to be confirmed]
:::

::: affiliations
^1^ Department of Applied Mathematics and Informatics, Fergana State University, Fergana, Uzbekistan; mirsaidbeky@gmail.com

^2^ Faculty of Physics and Mathematics, Fergana State University, Fergana, Uzbekistan; maxmaqsad@gmail.com

\* Correspondence: mirsaidbeky@gmail.com
:::

**Abstract:** Automatic recognition of Uzbek Sign Language is limited by the lack of public data and by evaluations in which the same signers appear in training and test sets. We address both problems with a compact pose encoder trained on Russian Sign Language and transferred to Uzbek. The encoder groups 85 MediaPipe keypoints per frame into four body-part tokens and models them over time with a Conformer; it has 3.4 million parameters and needs 0.76 billion floating-point operations per 32-frame window. We define a five-fold protocol with disjoint signers for the Slovo and Bukva datasets and compare models with paired bootstrap tests over signers. Under equal training conditions, the encoder reaches 70.8% top-1 accuracy on Slovo and 93.2% on Bukva, outperforming graph convolutional and Transformer baselines by 1.8–4.4 and 8.1–14.5 points, respectively. Ablations show that the body-part tokens carry the gain, whereas the skeletal graph can be removed without loss, reducing computation 3.9 times. On the official Slovo split, where 92 of 174 test signers also appear in training, the encoder matches the published accuracy of a video Transformer (64.1% versus 64.09%). On a pilot set of Uzbek fingerspelling with training budgets fixed in advance, pretraining on Russian Sign Language raises accuracy on an unseen participant from 24.7% to 47.5% and wins all 18 comparisons with training from scratch, whereas validation on recordings of the training participants overestimates accuracy by a median of 45 points. Signer-disjoint evaluation and cross-lingual pretraining are therefore essential for low-resource sign languages.

**Keywords:** sign language recognition; Uzbek Sign Language; Russian Sign Language; pose estimation; MediaPipe; graph convolutional network; Conformer; transfer learning; signer-independent evaluation; low-resource languages

# 1. Introduction

Automatic sign language recognition aims to turn the visual signal of a signed utterance into a symbolic representation that hearing-oriented software can process. Isolated sign language recognition (ISLR), in which each video contains one sign, is the simplest form of the task. It is also the building block of practical tools such as sign dictionaries with video search, learning applications and the vocabulary components of translation systems. For several sign languages, progress in ISLR has been driven by large public datasets: WLASL [@2020_li_wlasl], MS-ASL [@2019_joze_msasl], AUTSL [@2020_sincan_autsl] and ASL Citizen [@2023_desai_asl_citizen], and, for Russian Sign Language (RSL), Slovo [@2023_kapitanov_slovo], Bukva [@2024_kvanchiani_bukva] and Logos [@2025_ovodov_logos]. Most of the world's sign languages have no such resources: a recent survey of sign language datasets finds that public corpora are concentrated in a few high-resource languages, mostly from Europe, North America and East Asia, and lists no UzSL dataset [@2026_ni_sign_language_datasets_survey].

Uzbek Sign Language (UzSL) is one of these low-resource languages. The sign languages of the former Soviet republics are reported to be closely related to RSL because of the centralized language policy of the Soviet period; for Kazakh-Russian Sign Language this relation has been described by corpus developers, who also note that no formal comparison with RSL has been carried out [@2020_imashev_krsl_dataset; @2024_imashev_krsl_retrospective]. We are not aware of a formal linguistic comparison between UzSL and RSL either. The relation therefore motivates, but does not by itself justify, the use of RSL data for UzSL; in this work it is treated as a hypothesis that is tested empirically through transfer experiments. Published work on automatic UzSL recognition is recent and small in scale: it covers fingerspelling with hand landmarks or image detectors [@2026_jovlieva_webcam_uzbek_fingerspelling; @2024_kayumov_uzsl_latin_angular_signs; @2024_akhmedov_uzsl_dactyl_yolo] and isolated dynamic signs recorded from a few signers [@2026_ergashev_cnnlstm_vs_squeezeformer_uzsl; @2025_akkomron_uzslr_isolated_dynamic]. In these studies the data are not public, and the authors of at least one study state that their evaluation was not signer-independent [@2026_ergashev_cnnlstm_vs_squeezeformer_uzsl]. As a result, the reported accuracies are not comparable with each other, and it is unknown how they would hold for a signer the model has never seen.

Three obstacles stand between such studies and a recognizer that can be used by the Deaf community in Uzbekistan. The first is evaluation. A model tested on recordings of people who also appear in its training data measures memorization of individual signing styles as much as recognition of signs. Signer overlap is not limited to small in-house datasets: the authors of Slovo note that the signer groups of their training and test sets intersect, although they tried to reduce the intersection [@2023_kapitanov_slovo]. The second obstacle is deployment. Video networks such as MViTv2 [@2022_li_mvitv2] reach strong accuracy, but their size and computational cost make on-device use difficult. On-device keypoint estimators such as MediaPipe [@2020_zhang_mediapipe_hands; @docs_mediapipe_holistic_landmarker] turn an ordinary RGB camera into a low-cost body and hand sensor, so a compact model that works on keypoints is a natural choice for a mobile application. The third obstacle is data: a few hundred UzSL recordings cannot train a recognizer from scratch, and the obvious remedy, pretraining on a related and better-resourced sign language, has not been examined for UzSL.

This paper addresses the three obstacles together. We propose a compact pose encoder, the Multilingual Pose Encoder (MPE), which combines a skeletal graph over 27 keypoints with explicit body-part tokens and a Conformer [@2020_gulati_conformer] temporal encoder. We define a signer-disjoint five-fold evaluation protocol (SD-5) for Slovo and Bukva and use it, together with paired bootstrap tests over signers, for all model comparisons. We then pretrain MPE on RSL and transfer it to a pilot set of UzSL fingerspelling recorded for this study, with training budgets fixed before any pilot experiment was run.

The main contributions are as follows:

1. **A signer-disjoint evaluation protocol for two RSL datasets.** SD-5 splits Slovo and Bukva into five folds with no signer shared between training, validation and test data, and decides between models with a paired signer-level bootstrap. In the official Slovo split, 92 of the 174 test signers also appear in the training set, and their videos make up 67.6% of the test set (3,448 of 5,100); SD-5 removes this overlap. The fold lists and code are released.
2. **A compact pose encoder evaluated under equal training conditions.** MPE has 3.38 M parameters at inference and requires 0.76 GFLOPs per 32-frame window. It is compared with ST-GCN [@2018_yan_stgcn], SL-GCN [@2021_jiang_sam_slr], a keypoint Transformer and three simple baselines, all trained with the same step budgets and weight averaging. Under signer-disjoint evaluation MPE-A1 reaches 70.8 % top-1 on Slovo and 93.2 % on Bukva and exceeds the strongest baseline by 1.8 points on Slovo (ST-GCN) and by 8.1 points on Bukva (keypoint Transformer); the paired 95 % confidence intervals exclude zero. On the official Slovo split it matches the published accuracy of the MViTv2-S video network released with Slovo (64.1% versus 64.09%).
3. **An analysis of what drives accuracy.** Ablations show that explicit body-part tokens are the main structural element, whereas the skeletal graph is not needed: removing the part tokens lowers top-1 accuracy by 2.7 points on Slovo and 5.7 points on Bukva, while replacing the graph convolutions with a per-part MLP leaves Slovo accuracy unchanged (+0.0 points) and raises Bukva accuracy by 1.3 points (three seeds, paired 95 % confidence intervals); a three times larger variant (10.3 M parameters) adds 2.1 points on Slovo but gives no measurable gain on Bukva or on the Uzbek pilot.
4. **Russian-to-Uzbek transfer with pre-registered budgets.** Pretraining on RSL raises top-1 accuracy on an unseen UzSL participant from 24.7% (training from scratch) to 47.5%, pooled over the two test participants, and every pretrained condition outperforms training from scratch in all six seed–fold pairs (18 of 18 comparisons); a simulation on Bukva shows that with one example per class, RSL pretraining raises accuracy from 10.9% to 70.7%.
5. **A measurement of within-signer optimism.** On the pilot data, validation on held-out recordings of the training participants overestimates accuracy on a new participant by a median of 45.2 percentage points.

The rest of the paper is organized as follows. Section 2 reviews related work. Section 3 describes the data, the keypoint extraction, the evaluation protocol, the MPE architecture and the transfer setup. Section 4 reports the results, Section 5 discusses them together with the limitations of the study, and Section 6 concludes.

# 2. Related Work

## 2.1. Pose-Based Isolated Sign Language Recognition

Keypoint-based recognizers process sequences of body, hand and face landmarks instead of raw pixels. This reduces the input dimensionality, removes most of the background and appearance variation, and allows the same model to run on keypoints from different estimators and devices. Spatial-temporal graph convolutional networks (ST-GCN) [@2018_yan_stgcn] model the skeleton as a graph and introduced the partitioning of a node's neighbourhood into root, centripetal and centrifugal subsets, which is also used in this work. For sign languages, the SL-GCN model of the SAM-SLR framework [@2021_jiang_sam_slr] applies graph convolutions to a whole-body keypoint graph that includes both hands. Later work replaced or complemented graph convolutions with Transformers [@2017_vaswani_attention], added self-supervised pretraining of hand-aware pose models such as SignBERT+ [@2023_hu_signbert_plus], and used gloss semantics to support recognition [@2023_zuo_nla_slr]. Conformer blocks [@2020_gulati_conformer], which interleave self-attention with depthwise convolution, have recently been applied to signer-invariant continuous recognition [@2025_isharah_signer_invariant_conformer]. Our encoder follows the same family of designs but targets a different trade-off: a small model whose components can be ablated one by one under a strict signer-independent protocol.

## 2.2. Pretraining and Transfer Across Sign Languages

When target data are scarce, a model can be pretrained on a larger corpus and adapted to the target. OpenHands [@2022_selvaraj_openhands] released pose-based models and standardized pose datasets for several sign languages and studied pretraining across languages. SHuBERT [@2025_gueuwou_shubert] learns self-supervised representations from about 1,000 hours of American Sign Language video with separate streams for the hands, face and body, and Uni-Sign [@2025_li_uni_sign] unifies pretraining and fine-tuning for several sign language understanding tasks, with a pose-only setting in which the keypoints are divided into body parts. Closest to our setting, Logos [@2025_ovodov_logos] is a large RSL dataset whose pretrained encoder transfers to other sign languages, including few-shot settings with a frozen encoder. These studies transfer between large datasets of distant languages or rely on hundreds of hours of video. Our study examines the opposite end: transfer from public RSL datasets to a few hundred recordings of a closely related language, with an explicit protocol for fixing training budgets when no target validation data from new signers exist.

## 2.3. Datasets and Evaluation Protocols

Datasets differ not only in size but also in how signers are distributed across splits. AUTSL [@2020_sincan_autsl] and the ChaLearn LAP 2021 challenge built on it [@2021_sincan_chalearn_lap_islr] use signer-independent splits, and MS-ASL [@2019_joze_msasl] and ASL Citizen [@2023_desai_asl_citizen] also evaluate on signers who do not appear in the training data. For RSL, Slovo [@2023_kapitanov_slovo] contains 20,000 videos of 1,000 signs from 194 signers, and Bukva [@2024_kvanchiani_bukva] contains 3,757 videos of the 33 letters of the RSL manual alphabet from 155 signers. Both were recorded by many signers on various devices under varied conditions. The official Slovo split is not fully signer-disjoint [@2023_kapitanov_slovo], which makes it suitable for comparison with published results but not for measuring generalization to new signers. YouTube-ASL [@2023_uthus_youtube_asl] proposed a compact subset of 85 MediaPipe landmarks covering the upper body, both hands and the face, which we adopt as model input.

## 2.4. Uzbek Sign Language Recognition

Work on UzSL recognition has concentrated on the manual alphabet. One system recognizes Uzbek fingerspelling from a webcam using MediaPipe Hands, with a random forest for static letters and an LSTM for letters with movement, and reports 99.11% and 65.96% accuracy on data collected by the authors [@2026_jovlieva_webcam_uzbek_fingerspelling]; its authors note that an open UzSL dataset practically does not exist. Other studies classify Latin-script letters from finger-joint angles [@2024_kayumov_uzsl_latin_angular_signs] or detect letters in images with YOLO [@2024_akhmedov_uzsl_dactyl_yolo]. For isolated dynamic signs, Ergashev et al. compare a CNN-LSTM with a Squeezeformer-style Transformer on a closed keypoint dataset and report 87% and 91% accuracy, respectively, while stating that the split was not signer-independent [@2026_ergashev_cnnlstm_vs_squeezeformer_uzsl]. An open-source pipeline recognizes 50 dynamic signs recorded from 10 signers and a teacher with MediaPipe Holistic keypoints and a 1D-CNN–Transformer hybrid, but its data are not released [@2025_akkomron_uzslr_isolated_dynamic]. Thus, no UzSL dataset is publicly available, none of the published results was obtained on signers absent from training, and transfer from RSL has not been studied.

# 3. Materials and Methods

The pipeline is shown in Figure 1. Each video is converted into a sequence of 85 keypoints by MediaPipe Holistic Landmarker; the keypoints are normalized inside the model, encoded by MPE and classified. All models in this study use the same keypoint files, folds and evaluation code.

![**Figure 1.** Example fingerspelling signs from the UZSign.AI pilot: participants 002 and 003 (rows) and the letters L, M, O and SH (columns). Each frame is shown next to the same frame with the 85 MPE keypoints and the edges of the 27-node graph; long-range edges are dashed. Frames are taken from the middle of the longest run in which the dominant hand is detected.](figures/fig1_signs.png){width=16.5cm}


## 3.1. Datasets

Table 1 summarizes the three datasets.

**Slovo** [@2023_kapitanov_slovo] is an RSL dataset of isolated words: 20,000 videos of 1,000 signs recorded by 194 signers, plus 400 videos of a "no event" class. We excluded the "no event" class and used all 20,000 sign videos.

**Bukva** [@2024_kvanchiani_bukva] is an RSL fingerspelling dataset: 3,757 videos of the 33 letters of the RSL manual alphabet recorded by 155 signers. The videos are untrimmed and annotated with the start and end frames of the sign; we used only the frames between these boundaries. As in Slovo, the "no event" videos were excluded.

**UZSign.AI pilot** is the UzSL data recorded for this study with an in-house recording program at a resolution of 1280 × 720 pixels and 30 frames per second. It contains 322 videos of 40 fingerspelling signs: 29 letters of the Latin-script Uzbek alphabet, including the digraphs CH, SH and NG and the letters G' and O', and the numerals 0–10. Three participants took part. Participant 001 is the first author, who made three test recordings of one letter while checking the recording program; these recordings are kept in the training data of every fold and are never used for testing; participant 002 recorded 143 videos of 29 letters, and participant 003 recorded 176 videos of 24 letters and 11 numerals. The two main participants share 24 letters. Participants 002 and 003 were each recorded in a single session in January 2026, and the videos last from 0.30 to 3.67 s (mean 1.24 s). The recording program mirrors each frame before saving it; the mirroring was undone before keypoint extraction. All participants gave written informed consent to the recording, to the use of their data in research and to the publication of images in scientific articles. Because the videos contain the participants' faces and hands, they are personal data and are not shared (see the Data Availability Statement). [Ethics approval: name of the committee, protocol number and date — to be added by the authors.]

Table: **Table 1.** Datasets used in this study (after excluding "no event" videos).

| Dataset | Language | Content | Videos | Classes | Signers | Role |
|---|---|---|---|---|---|---|
| Slovo [@2023_kapitanov_slovo] | RSL | isolated words | 20,000 | 1,000 | 194 | model comparison; pretraining |
| Bukva [@2024_kvanchiani_bukva] | RSL | fingerspelling | 3,757 | 33 | 155 | model comparison; pretraining; few-shot simulation |
| UZSign.AI pilot (ours) | UzSL | fingerspelling and numerals | 322 | 40 | 3 | transfer target |

## 3.2. Keypoint Extraction and Quality Control

Keypoints were extracted with MediaPipe Holistic Landmarker 1.1.1 [@docs_mediapipe_holistic_landmarker] in video mode, with all detection, presence and tracking confidence thresholds set to 0.5. The same model file, identified by its SHA-256 hash, was used for all datasets. For each frame the landmarker returns 553 landmarks: 33 body pose points, 21 points for each hand and 478 face mesh points. Following YouTube-ASL [@2023_uthus_youtube_asl], we keep 85 of them: six pose points (shoulders, elbows and hips), the 2 × 21 hand points and 37 face points around the eyes, eyebrows, lips and face contour, including the two iris centres. Only the image coordinates *x* and *y* are used; the depth coordinate is discarded.

The landmarker reports whether each hand and the face are present in a frame but gives no per-point confidence for them. When a hand or the face is missing in some frames of a clip, its coordinates are linearly interpolated between the nearest frames where it was detected (or copied from the nearest detected frame at the clip boundaries), and a validity mask *m* is set to 0 for the filled frames. A part that is never detected in a clip is set to zero with *m* = 0.

To check extraction quality, we measured the hand detection rate: the share of frames in which a hand is detected among the frames in which the wrist of that arm is visible in the pose. This definition is conservative, because a hand resting at the lower edge of the frame counts as a miss even when its fingers are out of view. On the complete datasets the rate was 79.1% for Slovo (20,000 videos; 80.5% [78.8; 82.1] on average over the 194 signers, minimum 17.6%) and 70.2% for Bukva (3,757 videos; 73.2% on average over signers, minimum 31.2%); in Slovo the body pose was found in all frames and the face in 99.7% of them. The rate was 94.3% for the UzSL pilot (left hand 80.0%, right hand 96.0%). The body pose was detected in all frames of the pilot.

## 3.3. Signer-Disjoint Evaluation Protocol (SD-5)

All comparisons on Slovo and Bukva use a five-fold protocol with disjoint signers (Figure 2).

- **Slovo.** Six signers contribute most of the recordings, each with at least 500 videos; following a rule fixed before training, they are always assigned to the training data. The remaining 188 signers are divided into five groups, stratified by the number of recordings per signer, with a fixed random seed. In fold *k*, group *k* is the test set, 10% of the signers from the other groups form the validation set, and the rest are used for training. Over the five folds, each of the 188 signers, with 6,874 recordings in total, is tested exactly once.
- **Bukva.** All 155 signers are divided in the same way into five groups stratified by the number of recordings per signer.
- **Checks.** Before each run the code asserts that no signer and no video is shared between the training, validation and test sets, and that all keypoint files were produced by the same landmarker model.

![**Figure 2.** Signer-disjoint five-fold protocol (SD-5). (**a**) Fold assignment for Slovo: six signers with at least 500 videos each are always in the training data; the other 188 signers form five groups, each used once as the test set, and 10% of the signers of the other groups form the validation set. (**b**) Decision rule: a model is called better or worse only if the 95% bootstrap confidence interval of the paired difference over test signers excludes zero (the distributions are illustrative).](figures/fig2_protocol.png){width=16.5cm}


Each configuration is trained with three random seeds on each of the five folds. The primary metric is top-1 accuracy over all test recordings of the five folds, averaged over seeds. We also report top-5 accuracy, accuracy macro-averaged over test signers and the accuracy of the worst test signer among signers with at least ten test recordings. Confidence intervals (95%) are obtained by bootstrap resampling of test signers with 2,000 replicates [@2016_efron_hastie_casi]. A model is called better or worse than another only if the 95% confidence interval of the paired difference, computed over the same folds, seeds and resampled signers, excludes zero; otherwise we report that no difference was established.

The official Slovo split is used only to compare with published results. Its test set contains 5,100 videos of 174 signers, 92 of whom also appear in the official training set (3,448 test videos, 67.6%).

## 3.4. Multilingual Pose Encoder (MPE)

The architecture of MPE is shown in Figure 3. Its input is a clip of *T* = 32 frames with 85 keypoints per frame; its output is a *D*-dimensional embedding of the clip that is passed to a classification head.

![**Figure 3.** Architecture of the Multilingual Pose Encoder (MPE). (**a**) Overview: normalization, spatial encoder on the 27-node graph with joint and bone streams, body-part tokens, temporal encoder and attention pooling; solid arrows show the inference path and dashed arrows the training-only heads. (**b**) Graph convolution block with three neighbourhood subsets. (**c**) Conformer layer. (**d**) Classification head per dataset and auxiliary heads used during training.](figures/fig3_architecture.png){width=16.5cm}


### 3.4.1. Normalization

Normalization is part of the model, so the same computation is exported together with the network. Before any geometric operation, the *y* coordinates are multiplied by the frame height-to-width ratio *H*/*W* so that both axes share one scale. Each keypoint **p** is then described in two coordinate frames (Figure 4).

![**Figure 4.** Input channels of MPE for one frame of participant 003 (letter Gʻ): (**a**) video frame; (**b**) the 85 keypoints; (**c**) body frame normalized by the shoulder width; (**d**) local frames of the hands and the face (the left hand is not detected in this frame); (**e**) presence of the four body-part tokens over the 32-frame window.](figures/fig4_channels.png){width=16.5cm}

The body frame is shared by all 85 points:

$$\mathbf{x}^{\mathrm{body}} = \frac{1}{w}\,\mathbf{R}_{\theta}\,(\mathbf{p}-\mathbf{c}), \qquad (1)$$

where **c** is the midpoint of the shoulders, *w* is the shoulder width and **R**~θ~ rotates the shoulder line to the horizontal. To make the normalization robust to jitter, **c**, *w* and θ are averaged over the frames of the clip with weights given by the pose mask.

The local frame describes the shape of each hand and of the face independently of its position. For a hand,

$$\mathbf{x}^{\mathrm{loc}} = \frac{\mathbf{p}-\mathbf{p}_{0}}{\lVert \mathbf{p}_{9}-\mathbf{p}_{0} \rVert}, \qquad (2)$$

where **p**~0~ is the wrist and **p**~9~ is the base of the middle finger. For the face, the origin is the nose tip, the scale is the distance between the iris centres and the rotation aligns the line between the irises with the horizontal. For the six pose points, the local frame repeats the body frame. Scales are bounded from below by 10^−3^ and all coordinates are clipped to [−10, 10] to avoid numerical overflow when a part is degenerate. Each point is thus represented by six features, (*x*^body^, *y*^body^, *x*^loc^, *y*^loc^, *m*, *c*), where *m* is the validity mask and *c* is the confidence (pose visibility for pose points and the part presence flag for hand and face points).

### 3.4.2. Spatial Encoder on a 27-Node Graph

A subset of 27 points forms a graph (Figure 3). The body contributes seven nodes: the nose, the two iris centres, both shoulders and both elbows. Each hand contributes ten nodes: the wrist, the thumb base and tip, and the base and tip of the index finger, the base and tip of the middle finger, the tip of the ring finger, and the base and tip of the little finger (MediaPipe hand points 0, 1, 4, 5, 8, 9, 12, 16, 17 and 20). Body edges connect the nose with the irises and shoulders, the shoulders with each other, each shoulder with its elbow and each elbow with the wrist node of the corresponding hand. Hand edges follow the kinematic chain of the selected points. In addition, three long-range edges connect the two wrists and the tip of each index finger with the nose; they are intended to let information flow between parts that interact in signing (their effect is examined in Section 4).

Following ST-GCN [@2018_yan_stgcn], the neighbourhood of each node is split into three subsets (the node itself, neighbours closer to the centre and neighbours farther from the centre, with the nose as the centre), giving normalized adjacency matrices $\hat{\mathbf{A}}_k = \boldsymbol{\Lambda}_k^{-1/2}\mathbf{A}_k\boldsymbol{\Lambda}_k^{-1/2}$, *k* = 1, 2, 3. A graph convolution block computes

$$\mathbf{H}' = \mathrm{ReLU}\Big(\mathrm{BN}\Big(\sum_{k=1}^{3}\hat{\mathbf{A}}_k\,\mathbf{H}\,\mathbf{W}_k\Big) + \mathrm{Res}(\mathbf{H})\Big), \qquad (3)$$

where Res is the identity or a 1 × 1 projection when the number of channels changes. Three blocks with 64, 128 and *D* output channels operate on each frame separately. The encoder has two streams with identical structure and separate weights: a joint stream on the node features and a bone stream on the differences between each node and its parent in a spanning tree rooted at the nose. The two outputs are summed node by node.

### 3.4.3. Body-Part Tokens

The 27 node embeddings and the 58 points outside the graph are grouped into four part tokens per frame: left hand, right hand, body and face (Table 2). A token is the sum of three terms: the mask-weighted mean of the embeddings of its graph nodes; a two-layer MLP with GELU activations applied to the six features of its remaining points; and a learned embedding of the part type. A part that is absent in a frame is masked in the next step.

Table: **Table 2.** Composition of the body-part tokens.

| Token | Graph nodes | Points outside the graph | Total points |
|---|---|---|---|
| Left hand | 10 | 11 | 21 |
| Right hand | 10 | 11 | 21 |
| Body | 4 (shoulders, elbows) | 2 (hips) | 6 |
| Face | 3 (nose, iris centres) | 34 | 37 |
| **Total** | **27** | **58** | **85** |

### 3.4.4. Temporal Encoder, Pooling and Heads

In each frame, the four part tokens are fused by attention with a learned query and a softmax over the present parts, which gives one vector per frame; a learned positional embedding is then added. The sequence passes through a depthwise temporal convolution (kernel size 5, batch normalization, Swish activation, residual connection) and *N* Conformer layers [@2020_gulati_conformer]. Each Conformer layer consists of a half-step feed-forward module (expansion 4), multi-head self-attention with four heads and a frame mask, a convolution module (pointwise convolution, GLU, depthwise convolution with kernel size 15, batch normalization, Swish, pointwise convolution), a second half-step feed-forward module and layer normalization; the dropout rate is 0.1. Attention pooling over time produces the clip embedding,

$$\alpha_t = \operatorname{softmax}_t\big(\mathbf{w}^{\top}\tanh(\mathbf{W}\mathbf{h}_t)\big), \qquad \mathbf{z} = \sum_{t=1}^{T}\alpha_t\,\mathbf{h}_t . \qquad (4)$$

A linear classifier is attached to **z** for each dataset. When several datasets are used together, each example contributes a cross-entropy loss only through the head of its own dataset. Two auxiliary heads are used during training only: a normalized 256-dimensional embedding trained with cosine logits against learned class prototypes (scale 16), and one classifier per part token. The training loss is

$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_{e}\,\mathcal{L}_{\mathrm{emb}} + \beta \sum_{j=1}^{4}\mathcal{L}_{\mathrm{part},j}, \qquad (5)$$

with λ~e~ = 0.5 and β = 0.1.

### 3.4.5. Configurations

The base configuration (MPE-A0) uses *D* = 192 and *N* = 3. It has 4.41 M parameters during training and 3.38 M at inference, when the auxiliary heads are removed, and requires 0.76 GFLOPs per 32-frame window. FLOPs are counted with `torch.utils.flop_counter`, a multiply–add counting as two operations. The median CPU latency (float32, batch size 1, 10 warm-up runs followed by 100 timed runs) is 9.3 ms on an Intel Core i9-14900KF with 24 PyTorch threads on an otherwise idle machine; repeated measurements of the same architecture differ by 1–2 ms, so latencies are compared only by order of magnitude. The final configuration (MPE-A1) adds an exponential moving average (EMA) of the weights for validation and testing [@1992_polyak_averaging], with a decay of 0.999 for budgets of at least 3,000 steps and 0.99 otherwise (Slovo 0.999, Bukva 0.99); the averaged weights are updated at every step and used for validation, checkpoint selection and testing. EMA adds no parameters at inference. Larger variants used in the capacity study are described in Section 4.

## 3.5. Baselines and Training

All models receive the same keypoint files and are evaluated on the same folds. The baselines are as follows.

- **ST-GCN** [@2018_yan_stgcn] and **SL-GCN** [@2021_jiang_sam_slr] on the same 27-node graph and the same input features as MPE (3.41 M and 3.93 M parameters at inference; 1.98 and 2.03 GFLOPs).
- **Keypoint Transformer**: a pre-norm Transformer encoder [@2017_vaswani_attention] with a classification token that takes 67 points per frame (25 upper-body pose points and both hands, *x* and *y* normalized by the shoulders) and the presence masks, with *D* = 256, four layers and its own 64-frame window (3.53 M parameters at inference; 0.43 GFLOPs). It was trained with two recipes, the shared recipe described below and a recipe developed for it in our earlier experiments (a different temporal window and augmentation set, mirroring of left-handed signers and no auxiliary losses); the recipe was selected on validation data.
- **Simple baselines**: a two-layer Transformer, a bidirectional LSTM with attention pooling and a multilayer perceptron, trained with the shared recipe.
- **MViTv2-S** [@2022_li_mvitv2], the video model released with Slovo, evaluated only on the official Slovo split as a reference.

**Shared training recipe.** All keypoint models are trained with AdamW (learning rate 10^−3^, weight decay 0.05), linear warm-up over 5% of the steps followed by cosine decay to zero, no gradient clipping, label smoothing of 0.1, a batch size of 128 and bfloat16 mixed precision. The prototype-embedding loss (Section 3.4.4) is part of the shared recipe for all models, whereas part heads exist only in MPE. Training budgets are defined in optimizer steps and are identical for all models compared on a dataset: ⌈100 · *N*~train~ / 128⌉ steps, where *N*~train~ is the largest training set among the five folds, that is, 14,488 steps on Slovo and 2,154 on Bukva (about 100 epochs). The EMA of the weights is applied to all models in the main comparison except the three simple baselines, which are reported from earlier runs without it. During training, a 32-frame window is sampled with a random temporal rate between ×0.5 and ×2 and a random start; each frame is dropped with probability 0.1 and duplicated with probability 0.3, the sign boundaries are shifted by up to five frames (inwards only on Slovo and the pilot, in both directions on Bukva) while keeping at least half of the clip, a hand is removed in 20–25% of the frames, and the keypoints are randomly rotated (±15°), scaled (0.9–1.1), shifted (±0.05 of the frame width) and, for the hands, jittered with Gaussian noise (σ = 0.002 of the frame width). At test time, 32 frames are sampled uniformly over the whole clip. Validation accuracy is computed about 20 times per run and at the last step; the checkpoint with the best validation top-1 accuracy is tested once. Seeds 0, 1 and 2 control initialization, batch order and augmentation.

All models were trained on one NVIDIA GeForce RTX 5080 GPU (16 GB) with Python 3.14, PyTorch 2.14.1 (CUDA 13.2) and `torch.compile`; keypoints were extracted on the CPU with Python 3.12 and MediaPipe Holistic Landmarker 1.1.1, and models were exported with ONNX opset 17 and ONNX Runtime 1.31. Training on the GPU is not bit-exact deterministic; re-running the same configuration changes the results by about as much as changing the seed (Section 4). One MPE-A1 run takes about 4 min on Slovo.

## 3.6. Transfer to Uzbek Sign Language

### 3.6.1. Sources, Schemes and Folds

We compare three pretraining sources: Slovo, Bukva, and Slovo and Bukva together with separate classification heads. For transfer, the encoder is pretrained on all recordings of the source dataset, with 5% of the source signers held out for checkpoint selection. The target is the UZSign.AI pilot. Because participant 001 contributed only three recordings, it is always used for training, and the evaluation has two folds: the test participant is either 002 or 003, and the training data come from the other two participants. Some test classes do not occur in the training data of a fold (5 letters for participant 002 and the 11 numerals for participant 003), which caps the achievable accuracy at 82.5% and 67.0% of the test recordings, respectively.

Three schemes are compared: training from scratch, full fine-tuning and a linear probe. In full fine-tuning, the encoder is frozen for the first 20% of the steps and then unfrozen layer by layer every 7% of the steps, with a backbone learning rate of 0.1 times the head learning rate; in the linear probe the encoder stays frozen. The amount of target data is varied over one and two examples per class and all available recordings.

### 3.6.2. Budgets Fixed Before the Pilot Experiments

With only two or three participants, a validation set drawn from the training participants measures the wrong thing, and selecting checkpoints on it inflates the results (Section 4). We therefore fixed the training budgets before any pilot run, using a simulation on Bukva under SD-5 with four training recordings per class (132 recordings, validation on separate signers, five folds and three seeds). For each scheme, the budget in epochs is the median over the 15 runs of the step with the best validation accuracy: 1466.2 epochs for training from scratch and 209.5 epochs for fine-tuning and for the linear probe. On the pilot, the number of steps is ⌈epochs × *N*~train~ / 128⌉, where *N*~train~ is the number of training recordings in the fold; the model from the last step is evaluated, and no pilot data are used for model selection. The budgets were committed to version control before the first pilot run. Because the pilot runs are short (48–2,051 steps), the EMA decay is warmed up as min(0.99, (1 + *t*)/(10 + *t*)) at step *t*; otherwise the averaged weights would retain a large share of the random initialization of the new classification head. With all recordings, the validation recordings of the training participants are added to the training data; with one or two examples per class they are not, so that the number of examples per class is preserved.

Each pilot condition is run with three seeds on both folds, which gives six paired comparisons; pretraining is repeated for each seed. Because each fold has a single test participant, confidence intervals over signers cannot be computed, and a difference between two conditions is reported only if its sign is the same in all six pairs.

### 3.6.3. Few-Shot Simulation on Bukva

To study transfer at a controlled scale with many test signers, we also pretrain on Slovo and fine-tune on Bukva under SD-5 with 1, 3 and 10 training recordings per class and with all recordings. This simulation was run with MPE-A0, with checkpoints selected on the validation signers of each fold. About half of the Bukva test recordings come from signers who also appear in Slovo; we therefore also report accuracy on the recordings of the other signers only.

# 4. Results

All accuracies are top-1 percentages. On Slovo and Bukva they are pooled over the test recordings of the five folds and averaged over three seeds, with 95% bootstrap confidence intervals over test signers in brackets; a difference is called better or worse only if the confidence interval of the paired difference excludes zero (Section 3.3).

## 4.1. Comparison with Baselines under SD-5

Table 3 compares MPE-A1 with the baselines under SD-5. MPE-A1 reaches 70.8% on Slovo and 93.2% on Bukva and is better than every baseline on both datasets. On Slovo its margin over the graph convolutional networks is moderate, 1.8 points over ST-GCN and 2.7 points over SL-GCN, and 4.4 points over the keypoint Transformer. On Bukva the margins are larger: 8.1 points over the keypoint Transformer, 8.3 points over SL-GCN and 14.5 points over ST-GCN. The simple baselines trail by 8.6–42.2 points. The keypoint Transformer gained 3.9 points [+2.7; +5.2] on Slovo from its own recipe, which was selected on validation data, whereas the graph networks did not gain from it; each baseline is reported with the recipe that was better for it on validation. Applying the 64-frame window of the keypoint Transformer to MPE-A1 did not change its accuracy (+0.4 [−0.4; +1.1] on Slovo and −0.5 [−1.1; +0.1] on Bukva, one seed), so the margin is not an effect of the temporal window.

The accuracy of individual signers varies widely. On Slovo the mean accuracy over test signers is 69.9% for MPE-A1, but the weakest signer with at least ten test recordings reaches only 27.1%; on Bukva the corresponding values are 92.9% and 71.1%.

Table: **Table 3.** Top-1 accuracy under SD-5 (three seeds × five folds) and inference cost. Δ is the paired difference to MPE-A1 with its 95% confidence interval; all differences in the table exclude zero. Parameters, GFLOPs per window and CPU latency are given for the Slovo models (1,000 classes); the keypoint Transformer uses a 64-frame window, the other models a 32-frame window. \* Trained without weight averaging, shared recipe.

| Model | Params (M) | GFLOPs | CPU (ms) | Slovo top-1 [95% CI] | Δ Slovo | Bukva top-1 [95% CI] | Δ Bukva |
|---|---|---|---|---|---|---|---|
| **MPE-A1** | 3.38 | 0.76 | 9.3 | **70.8** [68.8; 72.7] | — | **93.2** [92.3; 94.0] | — |
| ST-GCN | 3.41 | 1.98 | 5.1 | 69.0 [67.0; 70.9] | −1.8 [−2.6; −1.1] | 78.7 [77.5; 79.9] | −14.5 [−15.6; −13.4] |
| SL-GCN | 3.93 | 2.03 | 19.2 | 68.1 [66.2; 69.9] | −2.7 [−3.4; −2.0] | 84.9 [83.9; 86.0] | −8.3 [−9.1; −7.4] |
| Keypoint Transformer | 3.53 | 0.43 | 1.8 | 66.3 [64.2; 68.4] | −4.4 [−5.7; −3.3] | 85.1 [82.8; 87.3] | −8.1 [−10.3; −6.1] |
| Two-layer Transformer\* | 1.98 | 0.11 | 1.7 | 56.4 [54.0; 58.6] | −14.4 [−15.8; −13.2] | 75.4 [72.9; 78.0] | −17.8 [−20.3; −15.2] |
| BiLSTM with attention\* | 11.00 | 0.62 | 5.2 | 52.3 [49.9; 54.7] | −18.4 [−19.7; −17.2] | 84.6 [83.3; 85.8] | −8.6 [−9.6; −7.6] |
| MLP\* | 1.55 | 0.004 | 1.0 | 28.6 [26.7; 30.5] | −42.2 [−43.7; −40.6] | 70.9 [69.2; 72.7] | −22.3 [−24.0; −20.6] |

Figure 5 relates accuracy to computation. MPE-A1 needs 2.6 times fewer floating-point operations than the graph networks, but on the CPU it is slower than ST-GCN (9.3 versus 5.1 ms). We attribute this to the many small operations of MPE (two streams, part tokens and the Conformer), which an eager PyTorch run on a desktop CPU executes less efficiently than the few large convolutions of ST-GCN; we did not investigate it further. Latencies of the same architecture differed by 1–2 ms between measurement sessions and are compared only by order of magnitude.

![**Figure 5.** Top-1 accuracy under SD-5 versus computation per window (logarithmic axis) on (**a**) Slovo and (**b**) Bukva. Filled blue circle: MPE-A1; open blue circles: MPE without the graph (MPE-A2) and the larger MPE from Section 4.2; grey squares: baselines. Error bars are 95% confidence intervals over test signers.](figures/fig5_cost.png){width=16.5cm}

## 4.2. What Drives Accuracy: Ablations and Capacity

Each ablation removes one element from MPE-A1 and is trained with the same three seeds and five folds (Figure 6).

- **Body-part tokens.** Replacing the four part tokens with a single token per frame lowers accuracy by 2.7 points [−3.3; −2.0] on Slovo and by 5.7 points [−6.9; −4.6] on Bukva. This is the only element whose removal hurts on both datasets.
- **Skeletal graph.** Replacing the graph convolution blocks with a per-part MLP over all points of the part, without the bone stream, leaves Slovo accuracy unchanged (+0.0 [−0.6; +0.7]) and improves Bukva accuracy by 1.3 points [+0.9; +1.8]. The validation data give the same picture (+0.1 [−0.8; +1.1] on Slovo and +0.9 [+0.6; +1.4] on Bukva). This graph-free variant, MPE-A2, has 3.14 M parameters at inference instead of 3.38 M and needs 0.196 GFLOPs per window, 3.9 times fewer; measured in one session, its CPU latency is 5.1 ms against 8.4 ms for MPE-A1.
- **Long-range edges.** Removing the wrist–wrist and index-tip–nose edges from the graph lowers Slovo accuracy by 0.9 points [−1.5; −0.3] but raises Bukva accuracy by 0.6 points [+0.2; +1.0]. On Slovo the graph without long-range edges is worse than no graph at all (69.9% versus 70.8%), so the long-range edges compensate for a limitation of the graph rather than adding information of their own.
- **Weight averaging.** Without EMA (MPE-A0), accuracy is 1.5 points lower on Slovo [−1.9; −1.0]; on Bukva the difference is not established (−0.3 [−0.7; +0.1]).
- **Capacity.** A larger MPE with *D* = 256 and *N* = 6 (10.27 M parameters, 1.33 GFLOPs) improves Slovo accuracy by 2.1 points [+1.6; +2.7], to 72.9%, but not Bukva accuracy (+0.2 [−0.1; +0.6]); on the UzSL pilot its difference to MPE-A1 had no consistent sign across the six seed–fold pairs.

The configuration of MPE-A1 was fixed before these ablations. Because the graph-free variant was identified on the test folds of SD-5, we report it as a lighter alternative rather than re-selecting the main model; its transfer to the pilot and its robustness are examined in Sections 4.4 and 4.6.

![**Figure 6.** Paired differences to MPE-A1 under SD-5 (three seeds × five folds) with 95% confidence intervals over test signers. Filled markers: the confidence interval excludes zero.](figures/fig6_ablations.png){width=16.5cm}

## 4.3. Official Slovo Split

Table 4 reports accuracy on the official Slovo split for comparison with the literature. Its test set contains 5,100 videos; 3,448 of them (67.6%) come from 92 signers who also appear in the training set, so these results partly reward familiarity with individual signers. MPE-A1 reaches 64.1% [61.3; 66.8], which matches the 64.09% reported for MViTv2-S, an RGB video network, by the authors of Slovo [@2023_kapitanov_slovo]. On the 966 test videos on which we ran the released MViTv2-S model ourselves, the paired difference between MPE-A1 and MViTv2-S is −0.2 points [−3.7; +3.1], so a difference is not established. In our MViTv2-S run the frame-sampling setting was chosen on the test videos, which can only favour MViTv2-S. The released MViTv2-S model has 35.0 M parameters (140.8 MB as an ONNX file) and processes RGB video; MPE-A1 has 3.38 M parameters with one classification head (13.5 MB as an ONNX file with three heads) and works on keypoints, so it is about ten times smaller. Weight averaging also helps on this split: MPE-A0 is 1.0 point below MPE-A1 (−1.0 [−1.5; −0.5]).

Table: **Table 4.** Top-1 accuracy on the official Slovo split (three seeds). Δ is the paired difference to MPE-A1.

| Model | Input | Test set | Top-1 [95% CI] | Top-5 | Δ to MPE-A1 |
|---|---|---|---|---|---|
| **MPE-A1** | keypoints | 5,100 videos | **64.1** [61.3; 66.8] | 86.6 | — |
| MPE-A0 | keypoints | 5,100 videos | 63.2 [60.2; 66.0] | 85.3 | −1.0 [−1.5; −0.5] |
| Keypoint Transformer | keypoints | 5,100 videos | 62.2 [59.6; 65.0] | 84.8 | −1.9 [−3.6; −0.1] |
| MViTv2-S, reported in [@2023_kapitanov_slovo] | RGB video | full test set | 64.09 | — | — |
| **MPE-A1** | keypoints | 966 videos | **64.9** [61.0; 68.5] | 87.5 | — |
| MViTv2-S, our run | RGB video | 966 videos | 65.1 [61.6; 68.7] | — | +0.2 [−3.1; +3.7] |

## 4.4. Transfer to Uzbek Sign Language

**Simulation on Bukva.** Figure 7a shows the few-shot simulation (MPE-A0, three seeds × five folds, checkpoints selected on the validation signers). Pretraining on Slovo raises accuracy from 10.9% to 70.7% with one training recording per class (+59.7 [+57.8; +61.6]), from 26.3% to 81.8% with three (+55.5 [+53.7; +57.6]) and from 63.7% to 86.8% with ten (+23.1 [+21.4; +24.7]). The linear probe is within 0.7 points of full fine-tuning at these steps. With all Bukva training data, pretraining on Slovo is slightly harmful: 90.9% against 92.9% from scratch (−2.0 [−2.8; −1.2]). Accuracy on the test signers who do not appear in Slovo is within 0.7 points of the values above.

**UzSL pilot.** Table 5 and Figure 7b give the pilot results for MPE-A1 with the budgets fixed in advance. Trained from scratch, the model reaches 9.5%, 11.7% and 24.7% with one example per class, two examples per class and all recordings. Every pretrained condition is better than training from scratch at the same step in all six seed–fold pairs, 18 of 18 comparisons, with mean gains of 15.3–31.3 points. For MPE-A1, the best pooled accuracy is obtained with Slovo and Bukva together and full fine-tuning: 40.3%, 41.2% and 47.5% at the three steps. The best source differs between the two test participants: with all recordings, pretraining on Bukva gives the best accuracy for participant 002 (50.3%) and pretraining on Slovo and Bukva together gives the best accuracy for participant 003 (45.6%). The linear probe is close to full fine-tuning (45.4% against 47.5% with all recordings). The graph-free MPE-A2 transfers as well as MPE-A1: with all recordings it reaches 47.9% with the linear probe and 45.8% with fine-tuning, and the sign of its difference to MPE-A1 is not constant at any step. Because some test classes do not occur in the training data of a fold, these accuracies are bounded by 82.5% for participant 002 and 67.0% for participant 003 (Section 3.6.1).

Table: **Table 5.** Top-1 accuracy on the UzSL pilot (MPE-A1 unless stated; three seeds × two folds): pooled over the two test participants, with participants 002 / 003 in parentheses. Every pretrained condition is better than training from scratch at the same step in all six seed–fold pairs.

| Pretraining | Scheme | One example per class | Two examples per class | All recordings |
|---|---|---|---|---|
| None | from scratch | 9.5 (12.8 / 6.8) | 11.7 (9.8 / 13.3) | 24.7 (21.0 / 27.7) |
| Slovo | linear probe | 31.0 (38.7 / 24.8) | 37.7 (42.7 / 33.7) | 41.2 (43.1 / 39.6) |
| Slovo | fine-tuning | 31.7 (35.2 / 28.8) | 33.9 (38.7 / 29.9) | 39.4 (42.0 / 37.3) |
| Bukva | linear probe | 27.5 (33.6 / 22.5) | 39.2 (45.0 / 34.5) | 46.1 (49.7 / 43.2) |
| Bukva | fine-tuning | 35.6 (40.6 / 31.6) | 38.5 (41.5 / 36.0) | 46.3 (50.3 / 43.0) |
| Slovo + Bukva | linear probe | 37.2 (46.4 / 29.7) | 39.4 (47.1 / 33.1) | 45.4 (49.9 / 41.7) |
| Slovo + Bukva | fine-tuning | **40.3** (48.5 / 33.7) | **41.2** (46.9 / 36.6) | **47.5** (49.9 / 45.6) |
| Slovo + Bukva, MPE-A2 | linear probe | 38.0 (44.8 / 32.6) | 40.3 (45.9 / 35.8) | 47.9 (52.7 / 43.9) |
| Slovo + Bukva, MPE-A2 | fine-tuning | 35.4 (36.1 / 34.8) | 40.5 (42.9 / 38.6) | 45.8 (47.1 / 44.7) |

![**Figure 7.** Transfer from RSL. (**a**) Simulation on Bukva under SD-5 (MPE-A0): top-1 accuracy with 1, 3 and 10 training recordings per class and with all recordings, from scratch and after pretraining on Slovo; error bars are 95% confidence intervals over test signers. (**b**) UzSL pilot (MPE-A1, full fine-tuning): top-1 accuracy pooled over the two test participants with one and two examples per class and with all recordings; vertical bars span the accuracies of participants 002 and 003.](figures/fig7_transfer.png){width=16.5cm}

## 4.5. Validation on the Training Participants Overestimates Accuracy

Before the budgets were fixed, our first pilot experiments selected checkpoints on held-out recordings of the training participants (MPE-A0). Figure 8 compares the validation accuracy at the selected checkpoint with the accuracy on the unseen participant for all 23 conditions of these experiments. Validation accuracy ranged from 54% to 100%, test accuracy from 7% to 51%, and the gap had a median of 45.2 points (range 35.0–75.1). The gap was largest for training from scratch, which can memorize the training participants without learning signs that generalize. A model that has seen a participant is nearly perfect on new recordings of that participant and loses half or more of its accuracy on a new one; this is why all pilot results in Section 4.4 use budgets fixed in advance and the last checkpoint.

![**Figure 8.** Validation accuracy on held-out recordings of the training participants (at the selected checkpoint) and test accuracy on the unseen participant for the 23 conditions of the early pilot experiments (MPE-A0), averaged over the two folds and sorted by the gap. k1, k2: one and two examples per class; all: all recordings; FT: full fine-tuning; probe: linear probe.](figures/fig8_valtest.png){width=16.5cm}

## 4.6. Robustness to Missing Hands and Keypoint Noise

Table 6 reports accuracy when the dominant hand is removed from a share *p* of the frames, as when the hand detector fails, and when Gaussian noise with standard deviation σ (as a fraction of the frame width) is added to the keypoints; the checkpoints of Section 4.1 are evaluated without retraining. Removing the dominant hand from a quarter of the frames does not change the accuracy of MPE-A1 (−0.1 [−0.4; +0.3] on Slovo and +0.1 [−0.2; +0.4] on Bukva), which is expected because a hand is dropped in 20–25% of the frames during training. Removing it from half of the frames costs 1.5 points on Slovo and 1.0 point on Bukva. Noise is the stronger factor: σ = 0.02 costs MPE-A1 6.4 points on Slovo and 4.2 points on Bukva. MPE-A2 is more robust to noise on both datasets, but on Slovo it is more sensitive than MPE-A1 to missing hands.

Table: **Table 6.** Top-1 accuracy of MPE-A1 and MPE-A2 under keypoint corruption (three seeds × five folds). *p*: share of frames without the dominant hand; σ: Gaussian noise as a fraction of the frame width. Δ: paired difference MPE-A2 − MPE-A1 with its 95% confidence interval.

| Corruption | Slovo A1 | Slovo A2 | Δ Slovo | Bukva A1 | Bukva A2 | Δ Bukva |
|---|---|---|---|---|---|---|
| None | 70.8 | 70.8 | +0.0 [−0.6; +0.7] | 93.2 | 94.5 | +1.3 [+0.9; +1.8] |
| *p* = 0.25 | 70.7 | 70.2 | −0.5 [−1.2; +0.2] | 93.3 | 94.0 | +0.7 [+0.3; +1.2] |
| *p* = 0.5 | 69.3 | 68.0 | −1.3 [−2.0; −0.5] | 92.2 | 93.0 | +0.8 [+0.3; +1.3] |
| σ = 0.01 | 70.0 | 69.8 | −0.2 [−1.0; +0.6] | 92.4 | 93.9 | +1.6 [+1.0; +2.1] |
| σ = 0.02 | 64.4 | 66.7 | +2.4 [+1.2; +3.6] | 89.0 | 92.4 | +3.4 [+2.7; +4.1] |
| *p* = 0.5, σ = 0.02 | 59.4 | 60.9 | +1.5 [+0.0; +3.1] | 87.6 | 90.5 | +2.9 [+2.2; +3.6] |

# 5. Discussion

**Part tokens matter more than skeletal topology.** The ablations point to one structural element: grouping the keypoints into body-part tokens with their own coordinate frames. Removing the tokens hurts on both datasets, most on Bukva, where the shape of one hand carries most of the information. The skeletal graph, in contrast, can be replaced by a simple MLP over the points of each part without loss, and with a gain on Bukva. A plausible explanation is that the local coordinate frames already make the hand shape explicit, so the MLP sees the whole hand at once, whereas the graph has to propagate information over several hops; interaction between parts is handled later by attention fusion and the Conformer. The long-range edges help only inside the graph and only on Slovo, which is consistent with this view. The practical consequence is that a pose encoder for sign recognition can be simplified: MPE-A2 keeps the accuracy of MPE-A1, is more robust to keypoint noise and needs 3.9 times fewer operations, at the cost of a somewhat larger sensitivity to missing hands on the word-level data.

**Compactness and speed.** In the number of operations MPE is cheaper than the graph networks, but on a desktop CPU it is not faster than ST-GCN. Floating-point operations are therefore an incomplete proxy for latency, and the latency of the final model has to be measured with the runtime and device on which it will run. We have not yet measured it on a phone.

**Transfer from RSL.** Both the simulation and the pilot show that pretraining on RSL is the main lever when target data are scarce: with one example per class, pretraining multiplied accuracy several times on Bukva and on the pilot. With all Bukva training data, pretraining on Slovo was slightly harmful, so the benefit shrinks and can reverse when enough target data are available. On the pilot, the better source differed between the two test participants, and the linear probe was close to full fine-tuning. A frozen encoder with a small trainable head is therefore a practical way to add new UzSL signs on a device. These results support the hypothesis that RSL representations transfer to UzSL fingerspelling, but they do not by themselves establish a linguistic relation between the two languages.

**Evaluation practice.** Two measurements in this study show how strongly signer overlap affects reported accuracy. On the pilot, validation on recordings of the training participants overestimated accuracy on a new participant by a median of 45 points; on the official Slovo split, two thirds of the test videos come from training signers. Accuracies obtained with signer overlap, including those reported for UzSL so far, should be read as upper bounds for unseen signers. The official split and SD-5 also differ in their training data and test signers, so their accuracies cannot be compared directly; comparisons between models should be made within one protocol.

**Limitations.** The UzSL pilot is small: two main participants, each recorded in one session, plus three test recordings of one letter by the first author; only fingerspelling and numerals; and a single recording setting. With one test participant per fold, differences on the pilot are judged by the sign of six paired comparisons, and confidence intervals over signers cannot be given. The ablations, the capacity study and the robustness tests were carried out on RSL data only. The robustness test uses independent synthetic noise and random frame dropping, which only approximate the temporally correlated errors of a real keypoint detector, and a change of keypoint extractor was not tested. Finally, the inference cost was measured on a desktop CPU, not on a mobile device.

**Future work.** The next steps are a multi-signer UzSL vocabulary dataset with signer-independent evaluation and human-verified labels, rejection of unknown or uncertain signs with coverage guarantees for new signers, short user enrollment for signer adaptation, and an on-device implementation with latency measured on phones.

# 6. Conclusions

This study shows that a compact keypoint-based encoder can be trained on Russian Sign Language and transferred to Uzbek Sign Language under an evaluation that never tests on signers seen during training. The proposed encoder outperforms graph convolutional and Transformer baselines trained under equal conditions, matches a video Transformer on the official Slovo split with a much smaller model, and owes its accuracy to body-part tokens rather than to the skeletal graph, which can be removed to reduce computation. Pretraining on Russian Sign Language consistently improves recognition of an unseen Uzbek participant, especially when only one or two examples per sign are available. Finally, validation on recordings of the training participants grossly overestimates accuracy on new signers, which argues for signer-disjoint protocols in future work on low-resource sign languages. The fold lists of the evaluation protocol and the code of the encoder are released to support such comparisons.
