## week 1 — ai/ml foundations

---

### setting up the environment

**prompt:**
"how do i install uv and set up a venv with it and then install numpy pandas and scikit-learn"

**result:**
got the commands, ran them, everything installed fine. venv is active and packages are working.

---

### loading mnist

**prompt:**
"give me the code to load mnist using fetch_openml and check the shape of X and y"

**result:**
got the fetch_openml call, printed X.shape and y.shape. X is (70000, 784) and y is (70000,). labels came back as strings which i didnt expect.

---

### fixing string labels

**prompt:**
"why are my mnist labels strings and how do i convert them to integers"

**result:**
got y.astype(int), turned it into encode_labels() function so its reusable and testable.

---

### writing the classifier

**prompt:**
"give me the code to train a random forest classifier on mnist and print accuracy precision and recall"

**result:**
got the full pipeline, ran it, accuracy came out around 0.97. added macro averaging for precision and recall since there are 10 classes.

---

### plotting the confusion matrix

**prompt:**
"how do i plot a confusion matrix using seaborn for a 10 class classifier and show the counts inside each cell"

**result:**
got sns.heatmap with annot=True and fmt="d". changed annot=False to True after seeing the plot was unreadable without numbers.

---

### extracting utility functions

**prompt:**
"which parts of my classifier.py code should i extract into separate functions to make it easier to unit test"

**result:**
got normalize_pixels, encode_labels and split_data as suggestions. made sense since each one does one thing and can be tested with dummy data without loading mnist.

---

### if __name__ == "__main__"

**prompt:**
"my tests were downloading mnist every time i ran pytest, how do i stop that"

**result:**
got the if __name__ == '__main__' fix. wrapped the pipeline in it and tests stopped triggering the download.

---

### setting up ruff

**prompt:**
"how do i install ruff and run it on my classifier.py file"

**result:**
ran ruff check classifier.py, it flagged a couple of unused import warnings. fixed them and got a clean pass.

---

### writing the unit tests

**prompt:**
"give me pytest unit tests for these three functions: normalize_pixels, encode_labels, split_data"

**result:**
got 6 tests total, two per function. checked range and exact values for normalize, dtype and values for encode, shapes and total count for split. all 6 passing.

### normalized confusion matrix

**prompt:**
"add a normalized confusion matrix, basically divide each row by its total so values are between 0 and 1 instead of raw counts. plot both the normal one and normalized one so we can compare"

**result:**
got a plot_confusion_matrix function with a normalize flag. when true it row divides by class totals. both versions plotted at the end, normalized one is way easier to read for spotting which digits get confused.

---

### class wise metrics

**prompt:**
"add classificationreport from sklearn to get precision recall and f1 for each digit separately instead of just the overall averages"

**result:**
got classification_report call after evaluation. can now see exactly which digits like 4 and 9 or 3 and 5 are harder to classify instead of just looking at one overall number.

---

### comparing multiple classifiers

**prompt:**
"compare multiple classifiers like decision tree, naive bayes, svm and random forest. train all of them and plot a barplot of their macro-f1 scores to see which one actually performs better instead of just guessing"

**result:**
got all four classifiers benchmarked on a 10k subsample since svm is too slow on full data. barplot sorted by macro-f1 with scores annotated on each bar. random forest came out on top which justified keeping it.

---

### hyperparameter tuning

**prompt:**
"add gridsearchcv to tune random forest hyperparameters like max_depth, n_estimators, min_samples_split etc. use cv=3 and macro-f1 as scoring metric then retrian the best model on full data"

**result:**
got gridsearchcv set up with the param grid and n_jobs=-1 to use all cores. best params printed after search then used to retrain on the full 56k training set. slight improvement in macro-f1 over default params.

