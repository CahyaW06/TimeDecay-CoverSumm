# EXTRACTIVE OPINION SUMMARIZATION USING COVERSUMM WITH TIME-DECAY WEIGHTING

[![License: MIT](https://img.shields.io/badge/License-MIT-green``.svg)](https://opensource.org/licenses/MIT)

We present the development of existing paper:

> [**Incremental Extractive Opinion Summarization Using Cover Trees**](https://arxiv.org/pdf/2401.08047),<br/>
[Somnath Basu Roy Chowdhury](https://www.cs.unc.edu/~somnath/)<sup>1</sup>, [Nicholas Monath](https://people.cs.umass.edu/~nmonath/)<sup>2</sup>, [Kumar Avinava Dubey](https://scholar.google.co.in/citations?user=tBbUAfsAAAAJ&hl=en)<sup>3</sup>, [Manzil Zaheer](https://scholar.google.com/citations?hl=en&user=A33FhJMAAAAJ)<sup>2</sup>, [Andrew McCallum](https://people.cs.umass.edu/~mccallum/)<sup>2</sup>, [Amr Ahmed](https://scholar.google.co.in/citations?user=tBbUAfsAAAAJ&hl=en)<sup>3</sup>, and [Snigdha Chaturvedi](https://sites.google.com/site/snigdhac/)<sup>1</sup>. <br>
UNC Chapel Hill<sup>1</sup>,  Google Deepmind<sup>2</sup>, Google Research<sup>3</sup>

## Overview

Extractive opinion summarization is the task of generating summaries from user reviews that can be very helpful in understanding an object. Extractive type summarization is necessary to avoid the hallucinations that typically occur in abstractive opinion summarization. CoverSumm, as a SOTA algorithm, performs efficiently for summarization in the incremental review environments often found in business review environments. However, CoverSumm treats all of the incoming reviews equally regardless of their age. This trait shows that CoverSumm still lacks in gathering time-relevant information. Hence, the summary can be biased by old reviews that are not relevant anymore. To address this gap, this paper proposes an enhanced version of CoverSumm that uses a time-decay weighting using exponential decay function. This optimization will prioritize recent reviews by recalculating their position in the new time-based centroid. We also replaced the use of SGTree data structure with Faiss’s IndexIDMap to handle the timestamp of each review with identical vector representation. We evaluated our method with the Solo Safari review dataset containing mixed languages, i.e., Indonesian, English, and Java. We use the recall metric from ROUGE Score as the main metric. The results show that Time-Decay CoverSumm significantly outperforms the baseline. Specifically, ROUGE-1 recall increases from 0.153 to 0.401, and ROUGE-L recall more than doubles from 0.073 to 0.153. Further analysis confirmed that our method successfully filtered out outdated topics and produced a summary that accurately reflected current opinion. This study shows that integrating temporal decay into centrality measurements is crucial for maintaining semantic relevance in dynamic opinion. Furthermore, this study can apply an extractive opinion summarization system in handling circulating reviews in the real case business sector.

## Installation
The simplest way to run our code is to start with a fresh environment.
```
conda create -n coverSumm python=3.8.20
conda activate coverSumm
pip install -r requirements.txt
```

## Summarization Algorithms

In this research, we compare the baseline algorithm CoverSumm with our proposed algorithm Time-Decay CoverSumm. The two algorithms used in the paper are available in the `src/algorithms/' folder.



## Dataset

We put our dataset in `data/` folder, so it can be used directly. We also put our code to get the dataset from our database in `data/generate.py`. You can adjust the code according to your needs.

<!-- ## Reference

```
@article{
        chowdhury2024incremental,
        title={Incremental Extractive Opinion Summarization Using Cover Trees},
        author={Somnath Basu Roy Chowdhury and 
                Nicholas Monath and 
                Kumar Avinava Dubey and 
                Manzil Zaheer and
                Andrew McCallum and
                Amr Ahmed and 
                Snigdha Chaturvedi
        },
        journal={Transactions on Machine Learning Research},
        year={2024},
        url={https://openreview.net/forum?id=IzmLJ1t49R},
}
``` -->
