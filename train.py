from __future__ import division
import sys
import re
import csv 
import math
import pickle

################################
#global variables
################################

##\boolean - set true to use NBC, false to use default 
use_nbc = False
##\boolean - set true to break nouns that appear in \\/
break_nouns = False
##\boolean - set true to change all words to lowercase
set_lower = False
##\list - 
misclas = []
##\int - values will never be smaller than exp(this)
very_low_exponent = -2000
##\boolean - keeps track if cross-validation is being made. False by default
cross_validation = False
##\boolean - whether the bigram tables will be written to a file
write_tables = False
##\boolean - whether the word and tag counts and sets will be written to a file
write_sets = False
##\int - i-th file to be added to test. 1 if no cross-validation is done
cross_val_selection = 1
##\list - Tags that can end a sentence
finishing_tags = ['.']
##\dict - tag-tag bigrams {(prev_tag, tag) : number of occurencerences}
dtag_tag = {}
##\dict - word-tag bigrams {(word, tag) : number of occurences}
dword_tag = {}
##\dict - tag occurences {tag : number of occurences}
count_tags = {}
##\dict - word occurences {word : number of occurences}
count_words = {}
##\set - set of words appearing in the dataset
words = set( [] )
##\set - set of tags appearing in the dataset
tags = set( ['START'] )
##\file - output file for statistics
f1score = open('f1.txt', 'w+')
##\file - output file for NBC training results
nf = open('nbc.txt', 'w+')
##\dict - saves the score for each stage of the Viterbi algorithm
score = {}
##\dict - saves the backpointer for each stage of the Viterbi algorithm
backpointer = {}
##\dict - saves the total probabilities for NBC
pi_c = {}
##\dict - saves the individual probabilities for NBC
theta_jc = {}
##\dict - saves the number of times each class of TAG has been used correctly on testing
correct = {}
##\dict -  saves the number of times each class of TAG has been used incorrectly on testing
incorrect = {}
##\dict - saves the number of times each class of tag has been used
test_tag_count = {}
##\dict - saves the accumulated accuracy for each class of tag
acc = {}
##\dict - saves the precision accuracy for each class of tag
prec = {}
##\dict - saves the F1-Score accuracy for each class of tag
f1dict = {}

################################
#add_dict
#adds one occurence to the bigram (prev, cur) count in dict with dict_id
#@args:
#	@ dict_id : 1 for tag-tag bigram dict, 2 for word-tag bigram dict
#	@ prev : previous tag in tag-tag, word in word-tag
#	@ cur : current tag in tag-tag, tag in word-tag
################################
def add_dict( dict_id, prev, cur ):
	#\tuple bigram
	key = ( prev, cur )
	if dict_id == 1:
		#if there is no object with the specific key, returns 0 by default
		val = dtag_tag.get( key, 0 )
		dtag_tag[key] = val + 1
	else:
		val = dword_tag.get( key, 0 )
		dword_tag[key] = val + 1

################################
#update_count
#adds one occurence to the tag or word in the corresponding dictionary
#@args:
#	@ dict_id : 1 for tag-tag bigram dict, 2 for word-tag bigram dict
#	@ element : the tag or word to increase the count for
################################
def update_count(dict_id, element):
	if dict_id == 1:
		val = count_tags.get( element, 0 )
		count_tags[element] = val + 1
	else:
		val = count_words.get( element, 0 )
		count_words[element] = val + 1

################################
#next_POS_filename
#builds and returns the name for the next file to be read
#@args:
#	@ folderpath : path of the current folder being read
#	@ i : index of the file to be read
################################
def next_POS_filename( folderpath, i):
	if i < 10:
		return folderpath + '0' + str( i ) + '.POS'
	elif i < 100:
		return folderpath + str( i ) + '.POS'

################################
#fill_tables
#controls the processing of each file in the dataset
#additionally, if @global cross_validation is set to True, selects the files for testing
#@args:
#	@ folderlist : filename with the names of all the folders of the dataset
################################
def fill_tables( folderlist ):
	global dtag_tag
	global dword_tag
	global misclas
	global cross_validation
	global tag
	global acc 
	global prec 
	global f1dict 
	
	test_filenames = []
	avg = 0

	for cross in range(0, 10):
		print 'Training fold ' + str(cross)
		dword_tag = {}
		dtag_tag = {}
		f = open( folderlist, 'r' )
		test_filenames = []
		for line in f:
			for i in range(0, 100):
				filename = next_POS_filename( line.strip(), i )
				if i % 10 == cross:
					test_filenames.append( filename )
					continue
				process_file( filename, True )

		if not cross_validation:
			#finish word-bigram training for each file and then test
			print('Finishing word-tag bigram training...')
			for filename in test_filenames:
				process_file( filename, False )

		print 'Training NBC...'
		train_nbc()
		nf.write('Fold ' + str(cross) + '\n')
		print 'Testing fold ' + str(cross)
		avg = avg + test( test_filenames )
		f.close()

	print 'Average error: ' + str( avg / len( misclas ) )
	for tag in tags:
		f1score.write('\nFor tag: ' + tag + '\nAccuracy: ' + str(acc.get(tag, 0)/10) + '\nPrecision: '+ str(prec.get(tag, 0)/10) + '\nF1: ' + str(f1dict.get(tag, 0)/10)+'\n')

################################
#coose_case
#controls the casing of known and unknown words
#@args:
#	@ word : word to be cased
#@return:
#	word in the correct format
################################
def choose_case( word ):
	global set_lower
	if set_lower:
		return word.lower()
	else:
		return word

################################
#test
#tests against a set of files
#@args:
#	@ test_filenames : test files to be used
################################
def test( test_filenames ):
	global finishing_tags
	global misclas
	global correct
	global incorrect 
	global count_tags
	global test_tag_count
	global acc
	global prec 
	global f1dict 

	total_tags = 0
	errors = 0
	test_tag_count = {}
	correct = {}
	incorrect = {}

	##For each test file
	for filename in test_filenames:
		f = open( filename, 'r' )
		pat = r'([\S]+)/([\S]+)'

		##Recovers all the contents of the file
		ff = f.read()
		f.close()
		res = re.findall( pat, ff )
		real_tags = [ ]
		prev_tag = 'START'
		begin = True
		sentence = ''

		##Process each pair of word and tags in the test file
		for pair in res:
			total_tags = total_tags + 1
			word = pair[0]
			tag = pair[1]
			
			##Controls the division of sentences in the test set
			if prev_tag == '.' and tag not in finishing_tags:
				inferred_tags = viterbi( sentence )

				##Compares each pair of inferred and real tags
				for inft, realt, wordt in zip(inferred_tags, real_tags, sentence.split(' ')):
					r = realt.split('|')

					if not inft in r:
						errors = errors + 1
						#Add an incorrectly used tag
						incorrect[inft] = incorrect.get(inft, 0) + 1
						test_tag_count[r[0]] = test_tag_count.get(r[0], 0) + 1
					else:
						#Add a correctly used tag
						correct[inft] = correct.get(inft, 0) + 1
						test_tag_count[inft] = test_tag_count.get(inft, 0) + 1

				begin = True

			if not begin:
				sentence = sentence + ' ' + word 
				real_tags.append( tag )
			else:
				sentence = word 
				real_tags = [ tag ]
				begin = False
				

			prev_tag = tag

	print "mistakes: " + str(errors) + " out of " + str(total_tags) + " = " + str(errors/total_tags) + "%"
	for tag in tags:
		if(correct.get(tag, 0) + incorrect.get(tag, 0) == 0):
			print 'Tag ' + tag + ' was tagged 0 times'
			accuracy = 1.0
		else:
			accuracy = correct.get(tag, 0) / (correct.get(tag, 0) + incorrect.get(tag, 0))
		
		if(test_tag_count.get(tag, 0) == 0):
			print 'Tag ' + tag + ' was found 0 times'
			precision = 1.0
		else:
			precision = correct.get(tag, 0) / test_tag_count.get(tag, 0)
		print 'Accuracy for ' + tag + ": " + str(accuracy)
		print 'Precision for ' + tag + ": " + str(precision)
		print 'F-1 for ' + tag + ": " + str((accuracy+precision)/2) + '\n' 

		acc[tag] = acc.get(tag, 0) + accuracy
		prec[tag] = prec.get(tag, 0) + precision
		f1dict[tag] = f1dict.get(tag, 0) + (accuracy+precision)/2


		if(precision > 1 or accuracy > 1):
			print 'Weird stuff going on... Found ' + str(test_tag_count.get(tag, 0)) + ' of ' + tag 
			print 'Correct: ' + correct.get(tag, 0) + ' and incorrect: ' + incorrect.get(tag, 0)
	misclas.append( errors / total_tags )
	return errors / total_tags

################################
#process_file
#updates the bigram tables and counters with the occurences of a particular file
#@args:
#	@ filename : of the current file to be read
################################
def process_file( filename, complete_training ):
	##recovers the global variables to be used inside this function
	global words
	global tags
	global finishing_tags
	global count_words
	global count_tags
	global break_nouns

	##Finds all the word-tag pairs in a file using regular expressions
	f = open( filename, 'r' )
	##This regular expressions finds all the strings with the shape: [non-space+]/[non-space+]
	pat = r'([\S]+)/([\S]+)'
	##Recovers all the contents of a file
	ff = f.read()
	res = re.findall( pat, ff )

	##Here begins the analysis of the word-tag pairs
	##The first tag of any file is START by default
	prev_tag = 'START'

	##For each word-tag pair
	for pair in res:
		##Handles the case in which the dataset contains a string with more than two slashes /
		if len( pair ) > 2:

			print("Exception found in " + filename + ": " + pair)
			continue

		else:
			##Handles exception for multiple tags separated by a vertical bar | in the dataset
			candidate_tags = pair[1].split('|')
			if len( candidate_tags ) > 1:
				word = choose_case( pair[0] )
				words.add( word )
				update_count( 2, word )
				##We keep the first tag by default, can be improved.
				tag = candidate_tags[0]
				##For each possible tag, we increase by 1 the occurence of the word with that tag
				for cand in candidate_tags:
					##Updates the word-tag bigram table
					add_dict(2, word, cand)

					##Updates the tag set and counts
					if complete_training:
						tags.add( cand )
						update_count( 1, cand )

			##Handles exception for words separated by /
			candidate_words = []
			if break_nouns:
				candidate_words = pair[0].split('\\/')
				if len( candidate_words ) > 1:
					tag = pair[1]

					##Updates the tag set and counts
					if complete_training:
						tags.add(tag)
						update_count(1, tag)

					##Each word's count is increased by one for the given tag
					for word in candidate_words:
						word = choose_case( word )

						##Updates the word-tag bigram table
						add_dict(2, word, tag)

						##Updates the word set and counts
						words.add( word )
						update_count(2, word)

			##Handles the normal case : one word with one tag
			if not( len( candidate_tags ) > 1 or len( candidate_words ) > 1 ):
				word = choose_case( pair[0] )
				tag = pair[1]

				##Updates the word set and counts
				words.add( word )
				update_count(2, word)
				##Updates the tag set and counts
				if complete_training:
					tags.add( tag )
					update_count(1, tag)
				##Updates the word-tag bigram table
				add_dict(2, word, tag)

			##Handles a finishing sentence: if the previous tag was a period and the new one is not...
			##(should handle three points)
			if prev_tag == '.' and tag not in finishing_tags:
				if complete_training:
					add_dict( 1, prev_tag, 'START' )
					update_count(1, 'START')
				prev_tag = 'START'

			##Updates the tag-tag bigram table
			if complete_training:
				add_dict( 1, prev_tag, tag )
			prev_tag = tag
	f.close()

################################
#print_word_tag
#writes the word-tag bigram table to a file
#@args:
################################
def print_word_tag():
	global dword_tag
	global tags
	global words 

	f = open('wordtag.txt', 'w+')

	line = ''
	for tag in tags:
		line = line + tag + '\t'
	f.write(line+'\n')
	line = ''

	for word in words:
		line = word + '\t'
		for tag2 in tags:
			key = (word, tag2)
			val = dword_tag.get(key, 0)
			line = line + str(val) + '\t'
		f.write(line + '\n')

	f.close()

################################
#print_word_tag
#writes the tag-tag bigram table to a file
#@args:
################################
def print_tag_tag():
	global dtag_tag
	global tags
	global words 

	f = open('tagtag.txt', 'w+')

	line = ''
	for tag in tags:
		line = line + tag + '\t'
	f.write(line+'\n')
	line = ''

	for tag in tags:
		line = tag + '\t'
		for tag2 in tags:
			key = (tag, tag2)
			val = dtag_tag.get(key, 0)
			line = line + str(val) + '\t'
		f.write(line + '\n')

	f.close()

################################
#write_tables_to_file
#writes the word and tag list and counts to files
#@args:
################################
def write_tables_to_file():
	global dtag_tag
	global dword_tag
	global count_tags
	global count_words
	global tags
	global words
	global pi_c
	global theta_jc

	f = open('dtt.txt', 'w+')
	pickle.dump(dtag_tag, f)
	f.close()

	f = open('dwt.txt', 'w+')
	pickle.dump(dword_tag, f)
	f.close()

	f = open('ct.txt', 'w+')
	pickle.dump(count_tags, f)
	f.close()

	f = open('cw.txt', 'w+')
	pickle.dump(count_words, f)
	f.close()

	f = open('tags.txt', 'w+')
	pickle.dump(tags, f)
	f.close()

	f = open('words.txt', 'w+')
	pickle.dump(words, f)
	f.close()

	f = open('pi.txt', 'w+')
	pickle.dump(pi_c, f)
	f.close()

	f = open('theta.txt', 'w+')
	pickle.dump(theta_jc, f)
	f.close()

################################
#load_tables()
#loads the tables of parameters from the stored files
#@args:
################################
def load_tables():
	global dtag_tag
	global dword_tag
	global count_tags
	global count_words
	global tags
	global words
	global pi_c
	global theta_jc

	dtag_tag = pickle.load( open('dtt.txt', 'r') )
	dword_tag = pickle.load( open('dwt.txt', 'r') )
	count_tags = pickle.load( open('ct.txt', 'r') )
	count_words = pickle.load( open('cw.txt', 'r') )
	tags = pickle.load( open('tags.txt', 'r') )
	words = pickle.load( open('words.txt', 'r') )
	#pi_c = pickle.load( open('pi.txt', 'r') )
	#theta_jc = pickle.load( open('theta.txt', 'r') )

################################
#calc_log_prob
#returns the natural logarithm of P(w_j|t_i)*P(t_i|t_k):
# ln(P(w_j|t_i)*P(t_i|t_k)) = ln(P(w_j|t_i)) + ln(P(t_i|t_k))
#@args:
#	@ word_j : current word
#	@ tag_i : current tag
#	@ tag_k : previous tag
################################
def calc_log_prob( word_j, tag_i, tag_k ):
	global dtag_tag
	global dword_tag
	global count_tags
	global count_words
	global tags
	global words
	global f2
	global use_nbc
	##Builds the keys to recover the count of both bigrams

	if not use_nbc:
		word_j = choose_word( word_j )
	key_word = ( word_j, tag_i )
	key_tag = ( tag_k, tag_i )

	##count(tag_i and tag_k)
	count_tag_tag = dtag_tag.get(key_tag, 0) + 1
	##count(word_j and tag_i)
	count_word_tag = dword_tag.get(key_word, 0) + 1
	##count(tag_i)
	count_tag_i = count_tags.get(tag_i, 0) + len( tags )
	##count(tag_k)
	count_tag_k = count_tags.get(tag_k, 0) + len( words )

	##count(tag_i and tag_k) / count(tag_k)
	p1 = math.log( count_tag_tag / count_tag_k )
	##count(word_j and tag_i) / count(tag_i)
	p2 = 0
	if use_nbc:
		if( word_j not in words ):
			#print 'Inferring tag for ' + word_j + ' using NBC'
			key = ( word_j, tag_i )
			x = dword_tag.get(key, -1)
			if( x == -1 ):
				p = classify_nbc( word_j )
				p2 = p[tag_i]
			else:
				p2 = x
		else:
			#print word_j + ' was found.'
			p2 = math.log( count_word_tag / count_tag_i )
	else:
		p2 = math.log( count_word_tag / count_tag_i )

	return p1 + p2

################################
#choose_word
#chooses a word (tentatively lowercase) or UNK if it hasn't been seen before
#@args:
#	@ word : word to be analysed
################################
def choose_word( word ):
	global words
	global use_nbc
	word = choose_case( word )

	if word not in words:
		return 'UNK'

	return word

################################
#viterbi
#chooses the best tag sequence for a given sentence
#@args:
#	@ sentence : to be POS-tagged
################################
def viterbi( sentences ):
	global very_low_exponent
	val = -1
	sentence = sentences.split(' ')
	##number of words in the sentence
	N = len( sentence )
	##number of tags
	K = len( count_tags )
	global tags

	##variables to keep the best tag and best value in a single iteration
	max_val = very_low_exponent
	max_tag = 'START'

	##Initialisation step
	for tag in tags:
		key = ( tag, 1 )
		
		score[key] = calc_log_prob( sentence[0], tag, 'START' )

		if( score[key] > max_val ):
			max_val = score[key]
			max_tag = tag
	
	prev_word = sentence[0]

	#Induction step
	for j in xrange(2, N+1):
		#word_j = choose_word( sentence[j-1] )

		for tag_i in tags:
			max_val = very_low_exponent
			max_k = 0
			max_log_prob = 0
			for tag_k in tags:
				prev_key = ( tag_k, j-1 )
				x = calc_log_prob( sentence[j-1], tag_i, tag_k )
				##We add here because we are using log probabilities
				val = score[prev_key] + x
				if( val >= max_val ):
					max_val = val 
					max_log_prob = val
					max_k = tag_k
			
			key = ( tag_i, j )
			if max_k == 0:
				print 'ERROR IN: ' + sentences + '\nat word: ' + str(j) + '\nval: ' + str(val) + '\nexp: ' + str(math.exp(val))
			score[key] = max_log_prob
			backpointer[key] = max_k

		prev_word = sentence[j-1]

	#Backtrace
	max_val = very_low_exponent
	max_k = 0
	t = []
	for tag in tags:
		key = ( tag, N )
		if( score[key] > max_val ):
			max_val = score[key]
			max_k = tag

	##t contains the inferred tags
	t.insert(0, max_k)
	max_val = 0

	try:
		for i in range( N-1, 0, -1 ):
			key = ( t[0], i + 1 )
			t.insert(0, backpointer[key])
	except KeyError as e:
		print 'KeyError ' + str(key) + ' not found'
		print 'Sentence:' + sentences 
		print 'Sequence so far:' + str(t)
		print 'N: ' + str(N) + '\ti: ' + str(i)
		exit(-1)

	#print( t )
	return t

################################
#unknown_words
#finds the probability distribution of unknown words using hapax legomena
#@args:
################################
def unknown_words():
	global words
	global tags
	global count_words
	global dword_tag

	unk_id = 'UNK'
	count_unk = 0
	##For each word that appears once...
	for word in words:
		if count_words[word] == 1:
			count_unk = count_unk + 1
			##Find out the tag for which it appears
			for tag in tags:
				key = (word, tag)
				val = dword_tag.get(key, 0)
				if val:
					key_unk = (unk_id, tag)
					dword_tag[key_unk] = dword_tag.get(key_unk, 0) + val
					update_count(1, tag)

	words.add( unk_id )
	count_words[unk_id] = count_unk		

################################
#isNumeric
#returns true if it contains at least one digit
#@args: 
#	@ token : the token to be tagged
################################
def isNumeric(token):
    return any([ch.isdigit() for ch in token])

################################
#get_features
#returns the vector of features for @word to be used in NBC
#@args:
#	@ word : 
################################
def get_features( word ):
	#Selected features:
	#Matches [alpha numeric]-[alpha numeric](-[alpha-numeric])+
	features = []
	lword = len( word )
	#Endings of 4 letters:
	if( lword >= 4 ):
		ending = word[-4:]
		if( ending == 'tion' ):
			features.append(1)
		else:
			features.append(0)

		if( ending == 'ness' ):
			features.append(1)
		else:
			features.append(0)

		if( ending == 'able' ):
			features.append(1)
		else:
			features.append(0)
	else:
		features.append(0)
		features.append(0)
		features.append(0)

	#Endings of 3 letters:
	if( lword >= 3 and word[-3:] == 'ing' ):
		features.append(1)
	else:
		features.append(0)

	#Endings of 2 letters:
	if( lword >= 2 ):
		ending = word[-2:]

		if( ending == 'ed' ):
			features.append(1)
		else:
			features.append(0)

		if( ending == 'ly' ):
			features.append(1)
		else:
			features.append(0)
	else:
		features.append(0)
		features.append(0)

	if( word[-1:] == 's' ):
		features.append(1)
	else:
		features.append(0)

	if( lword < 7 ):
		features.append(1)
	else:
		features.append(0)

	if( word.istitle() ):
		features.append(1)
	else:
		features.append(0)

	if( word.isupper() ):
		features.append(1)
	else:
		features.append(0)

	if( isNumeric( word ) ):
		features.append(1)
	else:
		features.append(0)

	if( word.isalpha() ):
		features.append(1)
	else:
		features.append(0)

	if( re.match('[\w](-[\w])+', word) ):
		features.append(1)
	else:
		features.append(0)

	return features

################################
#train_nbc
#finds the probability distribution of unknown words using NBC
#@args:
################################
def train_nbc():
	global words 
	global tags 
	global dword_tag
	global pi_c
	global theta_jc

	pi_c = {}
	theta_jc = {}
	N_c = {}
	N_jc = {}
	N = len(words)
	D = 13
	smoothing = 1

	for word in words:
		features = []
		tag = ''
		max_occur = 0
		for c_tag in tags:
			key = ( word, c_tag )

			if( dword_tag.get( key, 0 ) >= max_occur ):
				max_occur = dword_tag.get( key, 0 )
				tag = c_tag

		N_c[tag] = N_c.get(tag, 0) + 1
		features = get_features( word )

		for i in range(0, D):
			key = (i, tag)
			N_jc[key] = N_jc.get(key, smoothing) + features[i]

	for tag in tags:
		ct_tag = N_c.get(tag, smoothing)
		pi_c[tag] = ct_tag / N

		for i in range(0, D):
			key = (i, tag)
			theta_jc[key] = N_jc.get(key, smoothing) / (ct_tag + smoothing * len(tags)) 

################################
#logsumexp
#evaluates the logsumexp of p with maximum element B
#@args:
#	@ p : the probability distribution whose logsumexp we want to know
#	@ B : part of the logsumexp trick
################################
def logsumexp(p, B):
	global tags
	Z = 0
	for tag in tags:
		Z = Z + math.exp( p[tag] - B )

	return math.log( Z ) + B

################################
#classify_nbc
#returns the probability distribution of a particular word belonging to a specific tag class
#using NBC
#@args:
#	@ word : the word we want to analyse
################################
def classify_nbc( word ):
	global pi_c
	global theta_jc
	global dword_tag

	D = 13
	max_val = sys.maxint * -1
	y = 'NNP'
	p = {}

	for tag in tags:
		p[tag] = math.log( pi_c.get(tag, 0) )
		features = get_features( word )

		for i in range(0, D):
			key = (i, tag)
			if theta_jc.get(key) <= 0:
				print 'Can\'t take logarithm at:'
				print key 
				print 'Found: ' + str(theta_jc.get(key))
			if features[i]:
				p[tag] = p[tag] + math.log( theta_jc.get(key) )
			else:
				p[tag] = p[tag] + math.log( 1 - theta_jc.get(key) )

		if( max_val < p[tag] ):
			max_val = p[tag]
			y = tag

	nf.write( word + ' is ' + y +'\n')
	Z = logsumexp( p, max_val )
	for tag in tags:
		key = ( word, tag )
		p[tag] = p[tag] - Z
		dword_tag[key] = p[tag] 

	return p

################################
#main
#
#@args from command line:
#	0 - this file's name
#	1 - filename with folder names
#	2 - sentence to parse
################################
def main():
	global cross_validation
	global set_lower
	global write_tables
	global write_sets
	global break_nouns
	global use_nbc

	print 'Starting...'
	if len( sys.argv ) < 3:
		print "usage: python " + sys.argv[0] + " folderlist \"sentence\" [-cv] [-t] [-s]"
		sys.exit(-1)
	elif len( sys.argv ) > 3:
		flags = sys.argv[ 3: ]
		if "-cv" in flags:
			cross_validation = True
		if "-t" in flags:
			write_tables = True
		if "-l" in flags:
			set_lower = True
		if "-b" in flags:
			break_nouns = True
		if "-nbc" in flags:
			use_nbc = True

		print 'Filling bigram tables...'
		fill_tables( sys.argv[1] )

		print 'Calculating unknown words distribution...'
		unknown_words()

		if write_tables:
			print 'Saving tables in files...'
			write_tables_to_file()
	else:
		print 'Loading tables...'
		load_tables()


	print 'Predicting tags for new sentence...'
	t = viterbi( sys.argv[2] )
	print t
	print 'Finished.'
	nf.close()
	f1score.close()

if __name__ == '__main__':
  main()
