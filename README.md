# KaggleHomeDepot

This is a source code used by the team French Eagles for [Kaggle "Home Depot"](https://www.kaggle.com/c/home-depot-product-search-relevance) competition. It's also a project for Information Retrieval and Data Mining 2016 module at UCL. 

## Prerequisists:
- Python 2.7
- [nltk](http://www.nltk.org/)
- Java 1.7
- [Lucene](https://lucene.apache.org)
- [Opencsv](http://opencsv.sourceforge.net)

## How to use this code
1. Clone this repository
2. Create 'data' directory in project root directory as well as in AllenLucene
3. Download and unzip data files from Kaggle competition website to 'root' data directory.
4. Run before_lucene.py. It should create files AllenLucene/data/train.csv, AllenLucene/data/test.csv and directory AllenLucene/data/files with 124428 files.
5. Open AllenLucene project in IntelliJ IDEA or other Java IDE.
6. Include the follwing jars in your project: opencsv-3.7, lucene-queryparser-5.4.11, lucene-demo-5.4.11, lucene-core-5.4.11, lucene-analyzers-common-5.4.11.
7. Run IndexFiles.java. It should create directory AllenLucene/data/index.
8. Run SearchFiles.java. It should create files AllenLucene/data/lucene_train.csv and AllenLucene/data/lucene_test.csv.
9. Run preprocess_forum_stem.py. It should create files data/features_train.csv and data/features_test.csv.
10. Run learn.py. It should create file data/my_submission.csv.
11. Submit my_submission.csv file on kaggle and enjoy your result!


