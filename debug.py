import pandas as pd 

# Sample a small proportion for local experiment
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

def train_filter(patient_ids:str):
    if patient_ids in ["ID00007637202177411956430", "ID00010637202177584971671", "ID00011637202177653955184", "ID00012637202177665765362"]:
        return True
    return False

def test_filter(patient_ids:str):
    if patient_ids in ["ID00419637202311204720264", "ID00421637202311550012437"]:
        return True
    return False

small_train = train[train["Patient"].map(train_filter)]
small_test = test[test["Patient"].map(test_filter)]

small_train.to_csv("../data/small_train.csv", index=False)
small_test.to_csv("../data/small_test.csv", index=False)
