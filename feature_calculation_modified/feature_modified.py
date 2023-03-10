import re
import nltk
import stanza
import random
import collections
import numpy as np
from scipy.optimize import curve_fit
from lexical_diversity import lex_div as ld

from feature_calculation_modified.lexical_div_cal_modified import LexicalDiversity
from feature_calculation_modified.pos_ratios_cal_modified import HighLevelRatios
from feature_calculation_modified.dep_relations_modified import DependencyRelations

class Features:
	def __init__(self):
		self.nlp = stanza.Pipeline(lang='en', processors = 'tokenize , lemma, depparse , constituency, pos')
		self.optional_data = {}

	def safe_divide(self, numerator, denominator):
		if denominator == 0 or denominator == 0.0:
			index = 0
		else:
			index = numerator/denominator
		return index

	def get_all_features(self, sen_list):
		text = ' '.join(sen_list)
		temp  = {}

		#### Raw Features: Average Sentence Length and Standard Deviation of Sentence Length ####
		sen_len = [len(x.split()) for x in sen_list]
		temp['avg_sen_len'] = np.mean(sen_len)
		temp['std_sen_len'] = np.std(sen_len)

		#### Lexical Features: Character Diversity ####
		ld_measures = self.get_ld_measure(text)
		temp.update(ld_measures)

		#### POS Ratio Features and Lexical Features: Lexical Density ####
		POS = HighLevelRatios(text)
		pos_measures = POS.get_pos_ratios(text)
		temp.update(pos_measures)

		#### Syntactic Features: Frequency of Dependency Relations, Argument- Adjunct Ratio, Dependency Bigrams ####
		DEP = DependencyRelations()
		dep_measures = DEP.get_all_dep_feats(sen_list)
		temp.update(dep_measures)
		
		return temp

	def get_ld_measure(self, text):
		ld_text = ''.join(text.lower().split())
		ld = LexicalDiversity(ld_text)
		tokens = ld.get_tokens(char=True)
		ld_data = ld.get_lexical_measures(tokens)
		
		#scaling 
		ld_mean = {'TTR' : 8.11262665e-02, 
					'Root TTR' : 1.47188009e+00, 
					'Log TTR' : 5.64327689e-01, 
					'Maas TTR' : 1.68511737e-01,
					'Msstr' : 3.77196411e-01, 
					'Ma TTR' : 3.77330440e-01, 
					'HDD' : 4.21878311e-01, 
					'MTLD' : 1.43219078e+01,
					'MTLD MA' : 1.42771735e+01, 
					'MTLD MA Bi' : 1.42060573e+01, 
					'VocD' : 6.49950474e+00, 
					'YulesK': 6.02132971e+02
					}
		ld_std = {'TTR' : 3.86666604e-02, 
					'Root TTR' : 3.13960442e-01, 
					'Log TTR' : 4.11986152e-02, 
					'Maas TTR' : 6.58452413e-03,
					'Msstr' : 1.83366045e-02, 
					'Ma TTR' : 1.70646641e-02, 
					'HDD' : 1.96240139e-02, 
					'MTLD' : 8.88665331e-01,
					'MTLD MA' : 7.70135408e-01, 
					'MTLD MA Bi' : 7.66591821e-01, 
					'VocD' : 8.53453334e-01, 
					'YulesK' : 5.04564277e+01
					}
		
		for key, val in ld_data.items():
			ld_data[key] = (val - ld_mean[key])/ld_std[key]
		
		return ld_data