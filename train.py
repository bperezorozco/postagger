from __future__ import division
import sys
import re
import csv 
import math
import pickle

finishing_tags = {'.'}
dtag_tag = {}
dword_tag = {}
count_tags = {}
count_words = {}
score = {}
backpointer = {}
f2 = open('logprob.txt', 'w+')
words = set( [] )
tags = set( ['START'] )
sentences = set( [] )
exp = open( 'exp.txt', 'w+' )

#dict_id = 1 for TAG AND TAG
#dict_id = 2 for TAG AND WORD
def add_dict( dict_id, prev, cur ):
	key = ( prev, cur )
	if dict_id == 1:
		val = dtag_tag.get( key, 0 )
		dtag_tag[key] = val + 1
	else:
		val = dword_tag.get( key, 0 )
		dword_tag[key] = val + 1

def update_count(dict_id, element):
	if dict_id == 1:
		val = count_tags.get( element, 0 )
		count_tags[element] = val + 1
	else:
		val = count_words.get( element, 0 )
		count_words[element] = val + 1

def next_POS_filename( folderlist, i):
	if i < 10:
		return folderlist + '0' + str( i ) + '.POS'
	elif i < 100:
		return folderlist + str( i ) + '.POS'

def fill_tables( folderlist ):
	f = open( folderlist, 'r' )
	for line in f:
		for i in range(0, 100):
			process_file( next_POS_filename( line.strip(), i ) )

def process_file( filename ):
	f = open( filename, 'r' )
	pat = r'([\S]+)/([\S]+)'
	ff = f.read()
	res = re.findall( pat, ff )
	prev_tag = 'START'
	sentence = ''
	global words
	global tags
	global finishing_tags
	global count_words
	global count_tags

	for pair in res:
		if len( pair ) > 2:
			print pair 
			print filename
		else:
			candidate_tags = pair[1].split('|')
			candidate_words = pair[0].split('\\/')

#			Handles exception for | tags (multiple tags)
			if len( candidate_tags ) > 1:
				exp.write( str( str(pair) + ' in ' + filename + '\n' ) )
				word = pair[0].lower()
				words.add( word )
				update_count( 2, pair[0] )
				tag = candidate_tags[0]
				for cand in candidate_tags:
					add_dict(2, word, cand)
					tags.add( cand )
					update_count( 1, cand )

#			Handles exception for \/ words (multiple words)
			if len( candidate_words ) > 1:
				tag = pair[1]
				tags.add(tag)
				update_count(1, tag)
				for word in candidate_words:
					word = word.lower()
					add_dict(2, word, tag)
					words.add( word )
					update_count(2, word)

#			Handles the normal case 
			if not( len( candidate_tags ) > 1 or len( candidate_words ) > 1 ):
				word = pair[0].lower()
				tag = pair[1]

				words.add( word )
				update_count(2, word)
				tags.add( tag )
				update_count(1, tag)

				add_dict(2, word, tag)


			if ( prev_tag == '.' ) and ( tag not in finishing_tags ):
				sentences.add( sentence )
#				sentence = 'START ' + filename + ' ' + pair[0]
				sentence = pair[0]
				prev_tag = 'START'
				update_count(1, 'START')
	#			update_count(1, 'END')
			else:
				sentence = sentence + ' ' + pair[0]

			add_dict( 1, prev_tag, tag )
			prev_tag = tag

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

def write_to_file( wordsfile, tagsfile ):
	f = open( wordsfile, 'w+')
	global words

	for word in words: 
		f.write( str( word ) + '\n' )
	f.close()

	f = open( tagsfile, 'w+' )
	global tags
	for tag in tags:
		f.write( str( tag ) + '\n' )
	f.close()

	f = open( 'sentences.txt', 'w+' )
	global sentences
	for sencence in sentences:
		f.write( str(sencence) + '\n' )
	f.close()

	f = open( 'wordcount.txt', 'w+' )
	global count_words
	for word in words:
		f.write( word + ' ' + str(count_words.get(word, 0)) + '\n' )
	f.close()

	f = open( 'tagcount.txt', 'w+' )
	global count_tags
	for tag in tags:
		f.write( tag + ' ' + str(count_tags.get(tag, 0)) + '\n' )
	f.close()

def calc_log_prob( word_j, tag_i, tag_k ):
	global tags
	global f2
	key_word = ( word_j, tag_i )
	key_tag = ( tag_k, tag_i )

	count_tag_tag = dtag_tag.get(key_tag, 0) + 1
	count_word_tag = dword_tag.get(key_word, 0) + 1
	count_tag = count_tags.get(tag_i, 0) + len( tags )
	count_tag_start = count_tags.get(tag_k, 0) + len( words )

	
	f2.write(tag_i + ' after ' + tag_k + ': ' + str(count_tag_tag) + '\n')
	f2.write(word_j + ' as ' + tag_i + ': ' + str(count_word_tag) + '\n')
	f2.write('count ' + tag_i +': ' + str(count_tag) + '\n')
	f2.write('count ' + tag_k + ': ' + str(count_tag_start) +'\n')

	p1 = math.log( count_tag_tag / count_tag_start )
	p2 = math.log( count_word_tag / count_tag )

	f2.write('p1: ' + str(p1) + '\n' + 'p2: ' + str(p2) + '\n\n')
	return p1 + p2

def viterbi( sentence ):
	f = open('viterbi.txt', 'w+')
	sentence = sentence.split(' ')
	N = len( sentence )
	K = len( count_tags )
	global tags
	max_val = -1
	max_tag = 'START'

	#Initialisation step
	for tag in tags:
		key = ( tag, 1 )
		word = sentence[0].lower()
		score[key] = calc_log_prob( word, tag, 'START' )

		if( math.exp(score[key]) > max_val ):
			max_val = math.exp(score[key])
			max_tag = tag

		f.write( str(key) + ': ' + str(score[key]) + '\n')

	print( 'max tag: ' + max_tag + ' with prob:' + str((max_val)) )
	
	#Induction
	for j in xrange(2, N+1):
		word_j = sentence[j-1]
		prev_word = sentence[j-2]
		for tag_i in tags:
			max_val = 0
			max_k = 0
			max_log_prob = 0
			for tag_k in tags:
				prev_key = ( tag_k, j-1 )
				x = calc_log_prob( word_j, tag_i, tag_k )
				val = score[prev_key] + x
#				f.write(str(score[prev_key]) + ' + ' + str(x) + '=' + str(val)+'\n')
				f.write( sentence[j-1] + ' is tag ' + tag_i + ' if last tag was ' + tag_k + ' is ' + str(val) +'\n')
				if( math.exp( val ) > max_val ):
					max_val = math.exp( val )
					max_log_prob = val
					max_k = tag_k
#					f.write('new ' + str(max_val) + ' for ' + max_k + '\n')
#			print( str(score) )
			
			key = ( tag_i, j )
#			print('best tag for this round: ' + str(key) + ' ' + str(max_k))

			score[key] = max_log_prob
			backpointer[key] = max_k
			f.write(str(key) + ': ' + str(score[key]) + '\n')

	#Backtrace
	print(str(backpointer))
	max_val = 0
	max_k = 0
	t = []
	for tag in tags:
		key = ( tag, N )
		if( math.exp(score[key]) > max_val ):
			max_val = math.exp(score[key])
			max_k = tag

	t.insert(0, max_k)
	max_val = 0

	for i in range( N-1, 0, -1 ):
		key = ( t[0], i + 1 )
		t.insert(0, backpointer[key])

	vf = open('vit.txt', 'w+')
	pickle.dump(t, vf)
	vf.close()
	o = pickle.load( open('vit.txt', 'r') )
	print( o )

def main():
	print 'Starting...'
	args = sys.argv[ 1: ]
	sentence = 'i am very hungry .'
	print 'Filling tables...'
	fill_tables( args[0] )
	#process_file( r'WSJ-2-12/12/WSJ_1213.POS' )
	print 'Writing to files...'
	#write_to_file( args[1], args[2] )
	#print_tag_tag()
	#print_word_tag()
	viterbi(sentence)
	print 'Finished.'
if __name__ == '__main__':
  main()
