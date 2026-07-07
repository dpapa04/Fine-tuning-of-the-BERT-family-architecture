## Fine-tuning BERT models 

## Introduction

This README describes the process of developing and evaluating transformers as response clarity
classifiers based on the CLARITY dataset. The CLARITY dataset includes political question-
answer pairs where the target of the classifier is to assign one of three classes: Clear Reply,
Ambivalent, and Clear Non-Reply. The experiments focus on the fine-tuning of the following
three architectures: BERT (bert-base-uncased) and DistilBERT (distilbert-base-uncased).

## Data Handling and Preprocessing

First, the data was retrieved from Hugging Face datasets (ailsntua/QEvasion). Preliminary
exploratory data analysis indicated the presence of serious class imbalance, which can be ob-
served from the following class distribution:

• Ambivalent: 2,040 instances
• Clear Reply: 1,052 instances
• Clear Non-Reply: 356 instances

For the purpose of training transformer-based models, the question and answer from inter-
views were concatenated with the [SEP] token for each particular model. In addition to that,
to improve performance, a new feature was engineered: the normalized length of the answer.

Preprocessing Steps
• Tokenization using model-specific AutoTokenizers.
• Sequence padding and truncation to a MAX LEN of 512.
• Answer length calculation and normalization.

## Length Classifier

A custom architecture called LengthAwareClassifier was developed on PyTorch. The model
includes a pre-trained transformer and extends the classification head with the normalized
length feature of the answer. The training procedure followed the assignment’s requirement for
an explicit training loop, utilizing:
• Optimizer: AdamW
• Scheduler: Linear learning rate scheduler with warm-up
• Loss Function: CrossEntropyLoss (addressing multi-class classification)

## Experimental Setup

Some values that were tested during the trial and error included Epochs=3 (to avoid overfitting)
and variations in Batch Size, both of which negatively impacted the F1-score.
The following hyperparameters were selected based on standard practices and experimental
results:

Parameter       BERT / DistilBERT
Batch Size             16 
Learning Rate         2e-5
Epochs                  4
Seed                   158

## Predictive Performance

The performance of the models was evaluated using Accuracy and Macro F1-score:

Model                     Validation Accuracy     Validation Macro F1
BERT-base-uncased                 0.62                     0.58
DistilBERT-base-uncased           0.60                     0.62

## Conclusion
The experiments demonstrate that the DistilBERT model performs the best, as shown by the
F1-score on the leaderboard (DistilBERT got 0.62 while BERT got 0.58).
