from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()

x = digits.data
y = digits.target

split = train_test_split(x,y)

arrayClass = digits.target_names
arrayError = []
objectVote = {}
for class1 in arrayClass:
      objectVote[class1] = 0

datas = digits.data

for data1 in split[0] : 
      for data2 in data1 : 
            val = int(data2)
            if data2 in arrayClass:
                  objectVote[data2] += 1
            else:
                  arrayError.append(data2)
                  
tp = 0
fp = len(arrayError)
for vote in objectVote:
      tp += vote

precision = tp / (tp + fp)

print(objectVote)
print(tp)
print(fp)
print(precision)