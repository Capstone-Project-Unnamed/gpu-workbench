import pickle
import pprint

with open("case_0100.pkl", "rb") as f:
    data = pickle.load(f)

pprint.pprint(data)