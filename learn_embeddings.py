import word2vec
from nltk.tokenize import sent_tokenize, word_tokenize

textbooksFile = open('data/all.txt', 'r')
outFile = open('data/allTokenized.txt', 'w')
textbooks = textbooksFile.read()
print 'read!'

sentences = sent_tokenize(textbooks.decode('utf8'))
print 'sentence tokenization finished'

count = 0
outLines = list()
for s in sentences:
	count = count+1
	if count % 10000 == 0:
		print count
	tokens = word_tokenize(s)
	if len(tokens) < 3:
		continue
	outLines.append(str.join(' ', tokens))

print 'word tokenization finished'
outFile.write((str.join('\n', outLines)).encode('utf8'))

textbooksFile.close()
outFile.close()

#
# word2vec.word2phrase(
# 	'data/books/textbooks.txt', 'data/books/phrases', verbose=True)

print 'starting word2vec'
word2vec.word2vec(
	'data/allTokenized.txt', 'data/model_allTokenized.bin',
	 verbose=True, min_count = 5, threads = 4, size = 300, 
	 window = 9, iter_ = 10)
print 'finish'
