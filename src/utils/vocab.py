
import copy


class Vocab(object):
    def __init__(self):
        self.id2word = []
        self.word2id = {}

    def add(self, word):
        if word not in self.id2word:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, word):
        return self.word2id[word]

    def get_word(self, i):
        return self.id2word[i]

    def add_inverse(self):
        items = copy.deepcopy(self.id2word)
        for item in items:
            self.add('**'+item)

    @classmethod
    def load(cls, vocab_path):
        v = Vocab()
        with open(vocab_path) as f:
            for word in f:
                v.add(word.strip())
        return v


class RelationVocab(object):
    def __init__(self):
        self.id2word = []
        self.word2id = {}

    def add(self, word):
        if word not in self.id2word:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, word):
        return self.word2id[word]

    def get_word(self, i):
        return self.id2word[i]

    @classmethod
    def load(cls, vocab_path, inv_flg=False):
        v = Vocab()
        with open(vocab_path) as f:
            for word in f:
                v.add(word.strip())

        if inv_flg:
            words = copy.deepcopy(v.id2word)
            for word in words:
                v.add('**'+word)
        return v
