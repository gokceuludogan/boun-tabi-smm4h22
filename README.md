# BOUN-TABI@SMM4H'22: Text-to-Text Adverse Drug Event Extraction with Data Balancing and Prompting

This repository contains the implementation of the systems developed for the Social Media Mining for Health (SMM4H) 2022 Shared Task. We developed two separate systems for detecting adverse drug events (ADEs) in English tweets (Task 1a) and extracting ADE spans in such tweets (Task 1b). Our models rely on the T5 model and formulation of these tasks as sequence-to-sequence problems. To address the class imbalance, we used oversampling/undersampling on both tasks. For the ADE extraction task, we explored prompting to further benefit from the T5 model and its formulation. We built an ensemble model, which combines a model trained on oversampling/undersampling and another one trained with prompting. 

## Quick Usage

Our best performing models are available at Hugging Face! 

| **Model**                                                                               | **Description**                                                              |
|-----------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [t2t-assert-ade-balanced](https://huggingface.co/yirmibesogluz/t2t-assert-ade-balanced) | ADE identification model trained with over- and undersampled (balanced) data |
| [t2t-ner-ade-balanced](https://huggingface.co/yirmibesogluz/t2t-ner-ade-balanced)       | ADE extraction model trained with over- and undersampled (balanced) data     |
| [t2t-adeX-prompt](https://huggingface.co/gokceuludogan/t2t-adeX-prompt)                 | ADE extraction model trained with prompting                                  |

### ADE Identification 
```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("yirmibesogluz/t2t-assert-ade-balanced")
model = AutoModelForSeq2SeqLM.from_pretrained("yirmibesogluz/t2t-assert-ade-balanced")
predictor = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
predictor("assert ade: joints killing me now i have gone back up on the lamotrigine. sick of side effects. sick of meds. want my own self back. knackered today")
```

### ADE Extraction 

#### With Balancing
```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("yirmibesogluz/t2t-ner-ade-balanced")
model = AutoModelForSeq2SeqLM.from_pretrained("yirmibesogluz/t2t-ner-ade-balanced")
predictor = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
predictor("ner ade: i'm so irritable when my vyvanse wears off")
```
#### With Prompting

```python 
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("gokceuludogan/t2t-adeX-prompt")
model = AutoModelForSeq2SeqLM.from_pretrained("gokceuludogan/t2t-adeX-prompt")
predictor = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
predictor("Did the patient suffer from a side effect?: weird thing about paxil: feeling fully energized and feeling completely tired at the same time")
```
## Dataset

The dataset was provided by the organizers of the shared task and not publicly available. Once you obtained data by contacting the organizers, you may run the following script to preprocess it:


```bash
python prepare_data.py <data_dir> 
```

Data with prompt templates can be produced by:

```bash
python prompt_data.py <data_dir> 
```

## Training

After the required data sets are obtained, models can be trained with:

```bash
python train.py <task_prefix> <task_name> <input_data_dir> <model_output_dir>
```

where `<task_prefix` is `assert_ade` or `ner_ade` and `task_name` is either `smm4h_task1` for ADE classification or `smm4h_task2` for ADE extraction. 

## Ensemble

To ensemble model predictions, a config file is needed where models and output files are specified. See `config` directory for examples. Once a config file is ready, new predictions can be produced by:  

```bash
python ensemble.py <path-to-config-file>
```

## Results

### Task 1a: ADE Identification

| **Model**    | **Precision** | **Recall** | **F1**    |
|--------------|---------------|------------|-----------|
| BOUN-TABI    | **0.688**     | **0.625**  | **0.655** |
| SMM4H22 Mean | 0.646         | 0.497      | 0.562     |


### Task 1b: ADE Extraction

| **Model**    | **Partial Precision** | **Partial Recall** | **Partial F1** | **Strict Precision** | **Strict Recall** | **Strict F1** |
|--------------|-----------------------|--------------------|----------------|----------------------|-------------------|---------------|
| BOUN-TABI    | 0.507                 | **0.549**          | 0.527          | **0.384**            | **0.412**         | **0.398**     |
| SMM4H22-Mean | **0.539**             | 0.517              | 0.527          | 0.344                | 0.339             | 0.341         |


### Citation

```
@inproceedings{uludogan-yirmibesoglu-2022-boun,
    title = "{BOUN}-{TABI}@{SMM}4{H}{'}22: Text-to-Text Adverse Drug Event Extraction with Data Balancing and Prompting",
    author = {Uludo{\u{g}}an, G{\"o}k{\c{c}}e  and
      Yirmibe{\c{s}}o{\u{g}}lu, Zeynep},
    booktitle = "Proceedings of The Seventh Workshop on Social Media Mining for Health Applications, Workshop {\&} Shared Task",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.smm4h-1.9",
    pages = "31--34",
}
```

### Reference

```
@inproceedings{raval2021exploring,
  title={Exploring a Unified Sequence-To-Sequence Transformer for Medical Product Safety Monitoring in Social Media},
  author={Raval, Shivam and Sedghamiz, Hooman and Santus, Enrico and Alhanai, Tuka and Ghassemi, Mohammad and Chersoni, Emmanuele},
  booktitle={The 2021 Conference on Empirical Methods in Natural Language Processing},
  year={2021},
  organization={Association for Computational Linguistics (ACL)}
}
```

