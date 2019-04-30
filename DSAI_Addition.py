#!/usr/bin/env python
# coding: utf-8

# # 載入套件

# In[1]:


from keras.models import Sequential
from keras import layers
from keras import regularizers
import keras
import numpy as np
from six.moves import range


# # 加法器

# ### Parameters Config 參數設定

# 設定shell輸出文字的顏色。另外還能設定輸出文字的背景、底線、粗體等(參考:http://inpega.blogspot.com/2015/07/shell.html)

# In[2]:


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


# In[3]:


TRAINING_SIZE = 80000             #總樣本數
DIGITS = 3                        #所產生數字的最大位數
REVERSE = False                   #是否將數字從尾到頭反過來寫
MAXLEN = DIGITS + 1 + DIGITS      #兩個數字相加的字串最長長度
chars = '0123456789+ '            #訓練時會出現的所有字串
RNN = layers.LSTM                 #將RNN設定為LSTM
HIDDEN_SIZE = 128                 #隱藏層Neoren數
BATCH_SIZE = 128                  #Batch數
LAYERS = 1                        #層數


# #### 整理個別字串成dictionary形式/對字串做encoding/對字串做decoding

# In[4]:


class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)


# In[5]:


ctable = CharacterTable(chars)


# In[6]:


ctable.indices_char


# ### Data Generation 資料產生

# In[7]:


questions = []       #questions: 兩個數字相加
expected = []        #expected:  兩個數字相加的和
seen = set()         #檢查新產生的數字組合是否出現過
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    # 隨機產生最高三位數的數字(0~999)
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:                        #若新產生的數字組合已經出現過了，則不加入樣本中
        continue
    seen.add(key)
    q = '{}+{}'.format(a, b)               #將題目轉成字串形式
    query = q + ' ' * (MAXLEN - len(q))    #補上最後面的空格，使所有題目的字串長度一致
    ans = str(a + b)                       #將答案轉成字串形式
    ans += ' ' * (DIGITS + 1 - len(ans))   #補上最後面的空格，使所有答案的字串長度一致
    if REVERSE:                            #若REVERSE=TRUE，則將兩數相加的字串反過來寫
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))


# In[8]:


#檢查產生出的資料是否正確合理
print(questions[:5], expected[:5])


# ### Processing 資料前處理

# #### 將題目字串及答案字串重新編碼

# In[9]:


print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(expected), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)


# #### 將所有資料分成1/4的訓練資料和3/4的測試資料，在訓練資料中再分成90%的訓練集與10%的驗證集

# In[10]:


#隨機打散
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# train_test_split
train_x = x[:20000]
train_y = y[:20000]
test_x = x[20000:]
test_y = y[20000:]

#把training data的90%作為訓練集，10%作為驗證集
split_at = len(train_x) - len(train_x) // 10
(x_train, x_val) = train_x[:split_at], train_x[split_at:]
(y_train, y_val) = train_y[:split_at], train_y[split_at:]

#分別印出訓練集、驗證集及測試集的x和y的大小
print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Testing Data:')
print(test_x.shape)
print(test_y.shape)


# #### 查看輸入及輸出資料格式

# In[11]:


print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])


# ### Build Model 模型建立

# In[12]:


print('Build model...')

############################################
##### Build your own model here ############
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(DIGITS + 1))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

############################################

model.summary()


# ### Training 模型訓練

# In[13]:


for iteration in range(100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    #隨機取10筆樣本印出來，看有沒有回答正確
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)


# ### Testing 模型測試

# In[14]:


print("MSG : Prediction")
#####################################################
## Try to test and evaluate your model ##############
## ex. test_x = ["555+175", "860+7  ", "340+29 "]
## ex. test_y = ["730 ", "867 ", "369 "] 
right = 0
predictions = model.predict_classes(test_x, verbose = 1) 
for i in range(60000):
    correct = ctable.decode(test_y[i])
    guess = ctable.decode(predictions[i], calc_argmax=False)
    if correct == guess:
        right = right + 1

acc = right/60000
print("Accuracy : ", acc)
#####################################################


# Validation Accuracy為0.9826，Test Accuracy為0.9352，可以發現模型配得還不錯。

# ***

# # 減法器

# ### Parameters Config

# In[15]:


chars = '0123456789- '
ctable = CharacterTable(chars)
ctable.indices_char


# ### Data Generation

# #### 產生資料方式跟加法器相同
# #### 唯一差別是:由於規定a必須大於等於b，使相減後的差大於等於0，故將a小於b的組合作a,b互換的動作

# In[16]:


questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    #若a小於b，則令a、b互換
    if a < b :
        a = a + b
        b = a - b
        a = a - b
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}-{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a - b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total substraction questions:', len(questions))


# In[17]:


print(questions[:5], expected[:5])


# ### Processing

# In[18]:


print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(expected), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)


# In[19]:


indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# train_test_split
train_x = x[:20000]
train_y = y[:20000]
test_x = x[20000:]
test_y = y[20000:]

split_at = len(train_x) - len(train_x) // 10
(x_train, x_val) = train_x[:split_at], train_x[split_at:]
(y_train, y_val) = train_y[:split_at], train_y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Testing Data:')
print(test_x.shape)
print(test_y.shape)


# In[20]:


print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])


# ### Build Model

# In[21]:


print('Build model...')

############################################
##### Build your own model here ############
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(DIGITS + 1))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

############################################

model.summary()


# ### Training

# 模型在減法器的效果沒有很好，故將迭代次數調高至150

# In[22]:


for iteration in range(150):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)


# ### Testing

# In[23]:


print("MSG : Prediction")
#####################################################
## Try to test and evaluate your model ##############
## ex. test_x = ["555+175", "860+7  ", "340+29 "]
## ex. test_y = ["730 ", "867 ", "369 "] 
right = 0
predictions = model.predict_classes(test_x, verbose = 1) 
for i in range(60000):
    correct = ctable.decode(test_y[i])
    guess = ctable.decode(predictions[i], calc_argmax=False)
    if correct == guess:
        right = right + 1

acc = right/60000
print("Accuracy : ", acc)
#####################################################


# |TRAINING_SIZE|Iterarions|Validation Accuracy|Test Accuracy|
# |---|---|---|---|
# |80000|100|0.9682|0.8855|
# |80000|150|0.9799|0.9230|

# 由於迭代次數 = 100的模型準確率沒有很高，所以將迭代次數調高至150，準確率也因此提高。

# ***

# # 加法+減法

# ### Parameters Config

# 將原本的80000筆樣本分成40000筆加法和40000筆減法去建立模型與訓練會得到蠻低的準確率(約0.58)，故將樣本數增加至160000筆(80000筆加法和80000筆減法)

# In[24]:


TRAINING_SIZE = 160000
chars = '0123456789+- '
ctable = CharacterTable(chars)
ctable.indices_char


# ### Data Generation

# In[25]:


questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < (TRAINING_SIZE/2):
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)

while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    #若a小於b，則令a、b互換
    if a < b :
        a = a + b
        b = a - b
        a = a - b
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}-{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a - b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)

print('Total addition&substraction questions:', len(questions))


# ### Processing

# In[26]:


print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(expected), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)


# In[27]:


indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# train_test_split
train_x = np.concatenate([x[0:20000], x[80000:100000]])
train_y = np.concatenate([y[0:20000], y[80000:100000]])
test_x = np.concatenate([x[20000:80000], x[100000:160000]])
test_y = np.concatenate([y[20000:80000], y[100000:160000]])

#split_at = len(train_x) - len(train_x) // 10
(x_train, x_val) = np.concatenate([train_x[0:18000], train_x[20000:38000]]), np.concatenate([train_x[18000:20000], train_x[38000:40000]])
(y_train, y_val) = np.concatenate([train_y[0:18000], train_y[20000:38000]]), np.concatenate([train_y[18000:20000], train_y[38000:40000]])

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Testing Data:')
print(test_x.shape)
print(test_y.shape)


# In[28]:


print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])


# ### Build Model

# In[29]:


print('Build model...')

############################################
##### Build your own model here ############
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(DIGITS + 1))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

############################################

model.summary()


# ### Training

# In[30]:


for iteration in range(150):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)


# ### Testing

# In[31]:


print("MSG : Prediction")
#####################################################
## Try to test and evaluate your model ##############
## ex. test_x = ["555+175", "860+7  ", "340+29 "]
## ex. test_y = ["730 ", "867 ", "369 "] 
right = 0
predictions = model.predict_classes(test_x, verbose = 1) 
for i in range(60000):
    correct = ctable.decode(test_y[i])
    guess = ctable.decode(predictions[i], calc_argmax=False)
    if correct == guess:
        right = right + 1

acc = right/60000
print("Accuracy : ", acc)
#####################################################


# Validation Accuracy為0.9804，Test Accuracy為0.92355，可以看出模型的效果不錯，甚至表現得比只有減法的情形還好。

# # 乘法

# ### 最後來看一下同樣的模型在訓練乘法的資料是不是也會一樣效果不錯吧><

# ### Parameters Config

# In[32]:


TRAINING_SIZE = 80000
chars = '0123456789* '
LAYERS = 2


# In[33]:


class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)


# In[34]:


ctable = CharacterTable(chars)


# In[35]:


ctable.indices_char


# ### Data Generation

# In[36]:


questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}*{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a * b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)

print('Total multiplication questions:', len(questions))


# In[37]:


print(questions[:5], expected[:5])


# ### Processing

# In[38]:


print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(expected), DIGITS*2, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS*2)


# In[39]:


indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# train_test_split
train_x = x[:20000]
train_y = y[:20000]
test_x = x[20000:]
test_y = y[20000:]

split_at = len(train_x) - len(train_x) // 10
(x_train, x_val) = train_x[:split_at], train_x[split_at:]
(y_train, y_val) = train_y[:split_at], train_y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Testing Data:')
print(test_x.shape)
print(test_y.shape)


# In[40]:


print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])


# ### Build Model

# In[41]:


print('Build model...')

############################################
##### Build your own model here ############
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(DIGITS*2))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))
    
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

############################################

model.summary()


# ### Training

# In[42]:


for iteration in range(150):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)


# ### Testing

# In[43]:


print("MSG : Prediction")
#####################################################
## Try to test and evaluate your model ##############
## ex. test_x = ["555+175", "860+7  ", "340+29 "]
## ex. test_y = ["730 ", "867 ", "369 "] 
right = 0
predictions = model.predict_classes(test_x, verbose = 1) 
for i in range(60000):
    correct = ctable.decode(test_y[i])
    guess = ctable.decode(predictions[i], calc_argmax=False)
    if correct == guess:
        right = right + 1

acc = right/60000
print("Accuracy : ", acc)
#####################################################


# 
# |Layers|TRAINING_SIZE|Iterarions|Validation Accuracy|Test Accuracy|
# |---|---|---|---|---|
# |1|80000|150|0.5760|0.0425|
# |2|80000|150|0.6043|0.0492|

# 由上表中的Validation Accuracy及Test Accuracy可以看出模型在乘法的效果很不好，而從Test Accuracy小於Validation Accuracy很多也能推測模型可能有over fitting的問題。要改善模型可以從增加樣本數或更改模型的架構去著手。
