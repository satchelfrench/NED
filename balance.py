from cgi import test
import os
import pprint

import matplotlib.pyplot as plt
import seaborn as sns

## REQUIRES REFACTORING


# count the number of images in each class (case imbalance)
def set_balance(path):
    cases = next(os.walk(path))[1]
    d = {}
    for c in cases:
        classes = list(
                    filter(
                        lambda x: os.path.isdir(os.path.join(path, c, x)), 
                        os.listdir(os.path.join(path,c))
                    )
        )
        for cls in classes:
            echos = os.listdir(os.path.join(path,c,cls))
            l = len(echos)
            if d.get(cls) is None:
                d[cls] = l
            else:
                d[cls] += l
    return d



print('\nTraining set balances: ')
print('----------------------\n')
# train_bal = set_balance('../neo_echoset_full/')
train_bal = set_balance('./neo-echoset/')
pprint.pprint(train_bal)

x = list(range(0,len(train_bal.keys())))

x2 = [2*i for i in x]

plt.figure(figsize=(10,10))
# plot size, colours, etc.. 
# plt.bar(x2, train_bal.values(), width=1.8, color=['navy', 'darkorange', 'teal', 'fuchsia', 'lime','maroon', 'burlywood', 'aquamarine', 'tomato', 'violet', 'yellow', 'blue'])
plt.bar(x2, train_bal.values(), width=1.8, color=['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51','#606C38','#283618','#f4978e','#DDA15E','#BC6C25'])
plt.xticks(x2, labels=train_bal.keys(), rotation=90)
plt.ylabel('# of videos')
plt.subplots_adjust(bottom=0.25)
# plt.bar(train_bal.values(), height=100, label=train_bal.keys(), autopct='%.0f%%')
plt.show()


