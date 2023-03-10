import stanza

class HighLevelRatios:

    def __init__(self, text):
        self.text = text
        self.nlp = stanza.Pipeline(lang='en', processors = 'tokenize , lemma, depparse , constituency, pos')
        self.optional_data = {}

    def safe_divide(self, numerator, denominator):
        if denominator == 0 or denominator == 0.0:
            index = 0
        else:
            index = numerator/denominator
        return index
    
    def get_pos_ratios(self, text):
        doc = self.nlp(text)
        POS = {"adverb": 0, "adjective": 0, "pronoun":0 , "noun": 0, "verb": 0, "others": 0, "content_words": 0, "function_words": 0}
            
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == 'ADJ':
                    POS.adjective += 1
                if word.upos == 'ADV':
                    POS.adverb += 1
                if word.upos == 'PRON':
                    POS.pronoun += 1
                if word.upos == 'NOUN':
                    POS.noun += 1
                if word.upos == 'VERB':
                    POS.verb += 1
                else:
                    others += 1

        POS.content_words = sum([POS.noun, POS.verb, POS.adjective, POS.adverb])
        POS.function_words = sum([POS.pronoun, POS.others])
        
        pos_data = {'adverb/adjective' : self.safe_divide(POS.adverb, POS.adjective), 
                    'adverb/noun' : self.safe_divide(POS.adverb, POS.noun), 
                    'adverb/pronoun' : self.safe_divide(POS.adverb, POS.pronoun), 
                    'adjective/verb' : self.safe_divide(POS.adjective, POS.verb), 
                    'adjective/pronoun' : self.safe_divide(POS.adjective, POS.pronoun), 
                    'noun/verb' : self.safe_divide(POS.noun, POS.verb), 
                    'noun/pronoun' : self.safe_divide(POS.noun, POS.pronoun), 
                    'verb/pronoun' : self.safe_divide(POS.verb, POS.pronoun),
                    'content/function' : self.safe_divide(POS.content_words, POS.function_words)
                    }

        self.optional_data["POS"] = POS
        return pos_data