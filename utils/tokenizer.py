from pyknp import Juman

class JumanTokenizer:
    def __init__(self):
        self.analyzer = Juman()
    def tokenize(self, sentence):
        # This function is to return midashi from Jumanpp as tokens
        result = self.analyzer.analysis(sentence)
        tokens = []
        for mrph in result.mrph_list():
            tokens.append(mrph.midashi)
        return tokens