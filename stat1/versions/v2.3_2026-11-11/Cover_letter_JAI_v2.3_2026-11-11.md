# Cover letter — Journal of Automation and Intelligence (KeAi)

*Draft for the authors. Submission system: Editorial Manager (link on the journal page); editorial office: jai@cqu.edu.cn.*

---

Dear Editors,

We submit the manuscript **"Signer-disjoint evaluation and Russian-to-Uzbek transfer for isolated sign recognition: a multi-signer Uzbek Sign Language corpus and a compact pose encoder"** for consideration as a research article in the *Journal of Automation and Intelligence*.

Uzbek Sign Language (UzSL) has no public dataset, and every published recognition result for it was measured on signers who also appear in the training data. The manuscript gives the first results obtained on signers absent from training and shows what determines them:

- a multi-signer UzSL corpus of 5,993 isolated-sign recordings (2,381 classes, 12 signers) built from openly posted video lessons, labelled by the agreement of three independent cues and verified against a video dictionary (98.2% label accuracy on 113 reference pairs);
- leave-one-signer-out evaluation over ten signers: a compact keypoint encoder reaches 32.0% top-1 and 52.0% top-5 accuracy on 682 classes, reproduced on two external sources that took no part in corpus construction;
- cross-lingual pretraining on two public Russian Sign Language datasets as the main lever for a low-resource sign language (+14.7 points, a gain for every held-out signer, with training budgets fixed in advance by a pre-registered rule);
- a controlled comparison of keypoint models under a signer-disjoint five-fold protocol with paired bootstrap tests, which shows that released graph networks outperform our re-implementations of the same networks by 4–9 points, that body-part tokens rather than the skeletal graph carry the encoder's accuracy, and that validation on recordings of the training participants overestimates accuracy on a new signer by a median of 45 points.

All numbers in the manuscript are computed from the per-recording predictions by one analysis script released with the code; the fold lists, configurations and code are public (https://github.com/M1rsaiD/mpe-slovo-bukva), and the Supplementary Material gives per-class metrics, confusion lists and the full ablation sweep. We believe the paper fits the journal's scope on intelligent systems and machine learning applications, and that its evaluation discipline (no test signer ever seen in training, pre-registered decision rules, baselines run from released code) will be of use beyond sign language recognition.

The manuscript is original, has not been published previously and is not under consideration elsewhere. All authors have approved the manuscript and agree with its submission. The study involving human participants was approved retrospectively, after the pilot recordings had been made, by the Ethics Committee of the Faculty of Physics and Mathematics of Fergana State University (decision No. 12 of 9 November 2026), and all participants gave written informed consent. M. K. Onarkulov, a co-author and the Dean of the Faculty, did not take part in the committee's decision. The authors declare no competing interests. The use of a generative AI tool in the writing process is disclosed in the manuscript, as required by the journal's policy.

We would be grateful if the manuscript could be considered under the article processing charge waiver that the journal currently offers.

Corresponding author: Mirsaid A. Yusupov, Department of Applied Mathematics and Informatics, Fergana State University, Fergana, Uzbekistan; mirsaidbeky@gmail.com; ORCID 0009-0001-1989-0088.

Yours sincerely,

Maksadjon K. Onarkulov and Mirsaid A. Yusupov
