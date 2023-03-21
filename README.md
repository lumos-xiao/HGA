## Heuristic-Genetic-algorithm
## Datesets
There are three datasets used in our experiments:

- [IMDB](https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz)
- [AG's News](https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz)
- [Yahoo! Answers](https://s3.amazonaws.com/fast-ai-nlp/yahoo_answers_csv.tgz)

## Requirements
The code was tested with:

- python 3.6.5
- numpy 1.16.4
- tensorflow 1.8.0
- tensorflow-gpu 1.5.0
- pandas 0.23.0
- keras 2.2.0
- scikit-learn 0.19.1
- scipy 1.0.1

## File Description

- `textcnn.py`: one of the attacked models.
- `train_orig.py`: Training models with original database. 
- `heuristic_Genetic_algorithm.py` : Attacking the models by the heuristic genetic algorithm (HGA).
