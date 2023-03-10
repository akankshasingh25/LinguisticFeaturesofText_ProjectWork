import re
import nltk
import stanza

class DependencyRelations:
	def __init__(self):
		self.nlp = stanza.Pipeline(lang='en', processors = 'tokenize , lemma, depparse , constituency, pos')
	
	def safe_divide(self, numerator, denominator):
		if denominator == 0 or denominator == 0.0:
			index = 0
		else:
			index = numerator/denominator
		return index
	
	def get_dep_rel(self, sentence):
		doc = self.nlp(sentence)
		dep = {}
		for sent in doc.sentences:
			for word in sent.words:
				if word.deprel != 'root':
					if word.deprel in dep:
						dep[word.deprel] += 1
					else:
						dep[word.deprel] = 1

		return dep

	def get_dependency_features(self, sentence):
		doc = self.nlp(sentence)
		bigrams = []
		relations = []
		for index,sent in enumerate(doc.sentences):
			for word in sent.words:
				if word.deprel != 'root':
					relations.append(word.deprel)
					if word.id > word.head: 
						position = 'before'
					else:
						position = 'after'
					feat = str((doc.sentences[index].words[word.head-1].upos, word.upos, position))
					bigrams.append(feat)
		rel_big = nltk.FreqDist(relations)
		dep_big = nltk.FreqDist(bigrams)
		return rel_big, dep_big

	def get_all_dep_feats(self, sen_list, text=None):
		dep_feat_data = {}
		arguments_tags = ['nsubj', 'obj', 'ccomp', 'conj', 'csubj:pass', 'iobj']
		
		#dep_rel_list = []
		#dep_big_list = []

		for sen in sen_list:
			sen = re.sub(r'[^\w\s]', ' ', sen)
			sen = ' '.join(sen.split())
			rel_big, dep_big = self.get_dependency_features(sen)
			adjuncts = sum([rel_big[x] for x in list(rel_big.keys()) if x not in arguments_tags])
			arguments = sum([rel_big[y] for y in list(rel_big.keys()) if y in arguments_tags])
			dep_feat_data['arguments/adjuncts'] = self.safe_divide(arguments, adjuncts)
			
			for key, val in (rel_big + dep_big).items():
				if key in dep_feat_data:
					dep_feat_data[key] += val
				else:
					dep_feat_data[key] = val

			#dep_rel = self.get_dep_rel(sen)
			#dep_rel_list.append(dep_rel)

			#dep_big_list.append(dep_big)

		return dep_feat_data