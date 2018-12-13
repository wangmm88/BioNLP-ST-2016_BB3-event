# -*- coding: utf-8 -*-
import kindred
# from Token import Token
# from Sentence import Sentence
from intervaltree import IntervalTree
from collections import defaultdict
import six
class Parser:
    """
    Runs Spacy on corpus to get sentences and associated tokens
    :ivar language: Language to parse (en/de/es/pt/fr/it/nl)
    :ivar nlp: The underlying Spacy language model to use for parsing
    """
    _languageModels = {}
    def __init__(self, language='en'):
        """
        Create a Parser object that will use Spacy for parsing. It offers all the same languages that Spacy offers. Check out: https://spacy.io/usage/models. Note that the language model needs to be downloaded first (e.g. python -m spacy download en)
        :param language: Language to parse (en/de/es/pt/fr/it/nl)
        :type language: str
        """
        # We only load spacy if a Parser is created (to allow ReadTheDocs to build the documentation easily)
        import spacy
        acceptedLanguages = ['en', 'de', 'es', 'pt', 'fr', 'it', 'nl']
        assert language in acceptedLanguages, "Language for parser (%s) not in accepted languages: %s" % (
            language, str(acceptedLanguages))
        self.language = language
        if not language in Parser._languageModels:
            # Parser._languageModels[language] = spacy.load(
                # language, disable=['ner'])
            Parser._languageModels[language] = spacy.load(language)
        self.nlp = Parser._languageModels[language]
    def _sentencesGenerator(self, text):
        if six.PY2 and isinstance(text, str):
            text = unicode(text)
        # TODO: add better sentence segmenter
        text = text.strip().replace("\n", " ").replace("\r", " ")
        text = text.lower()
        parsed = self.nlp(text)
        sentence = None
        # print(parsed)
        # num = 4
        # print(parsed[0].text, parsed[0].ent_iob_, parsed[0].ent_type_)
        # print(parsed[num].text, parsed[num].ent_iob_, parsed[num].ent_type_)
        for token in parsed:
            if sentence is None or token.is_sent_start:
                if not sentence is None:
                    yield sentence
                sentence = []
            sentence.append(token)
        if not sentence is None and len(sentence) > 0:
            yield sentence
    def parse(self, corpus):
        """
        Parse the corpus. Each document will be split into sentences which are then tokenized and parsed for their dependency graph. All parsed information is stored within the corpus object.
        :param corpus: Corpus to parse
        :type corpus: kindred.Corpus
        """
        assert isinstance(corpus, kindred.Corpus)
        # Ignore DeprecationWarning from SortedDict which is inside IntervalTree
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # for d in corpus.documents[:2]:
        PAD_TOKEN = '<PAD>'
        UNK_TOKEN = '<UNK>'
        tagdict = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        depdict = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for d in corpus.documents:
            entityIDsToEntities = {
                entity.entityID: entity for entity in d.entities}
            denotationTree = IntervalTree()
            entityTypeLookup = {}
            for e in d.entities:
                entityTypeLookup[e.entityID] = e.entityType
                for a, b in e.position:
                    if b > a:
                        denotationTree[a:b] = e.entityID
                    else:
                        raise Exception
            # print(denotationTree)
            for sentence in self._sentencesGenerator(d.text):
                # print(sentence)
                tokens = []
                for t in sentence:
                    ent_type_ = 'O' if t.ent_type_ == '' else t.ent_type_  
                    token = kindred.Token(t.text, t.lemma_, t.tag_, ent_type_,
                                  t.idx, t.idx+len(t.text))
                    tokens.append(token)
                    if t.tag_ not in tagdict:
                        tagdict[t.tag_] = len(tagdict)
                
                sentenceStart = tokens[0].startPos
                sentenceEnd = tokens[-1].endPos
                sentenceTxt = d.text[sentenceStart:sentenceEnd]
                indexOffset = sentence[0].i
                dependencies = []
                # print(tokens)
                for t in sentence:
                    depName = t.dep_
                    if depName == 'ROOT':
                        dep = (0, t.i-indexOffset, depName)
                    else:
                        dep = (t.head.i-indexOffset+1, t.i-indexOffset, depName)
                    dependencies.append(dep)
                    # add depdict dict
                    if depName not in depdict:
                        depdict[depName] = len(depdict)
                        
                # print(tokens)
                # print(dependencies)
                entityIDsToTokenLocs = defaultdict(list)
                for i, t in enumerate(tokens):
                    entitiesOverlappingWithToken = denotationTree[t.startPos:t.endPos]
                    for interval in entitiesOverlappingWithToken:
                        entityID = interval.data
                        entityIDsToTokenLocs[entityID].append(i)
                # print(entityIDsToTokenLocs)
                sentence = kindred.Sentence(
                    sentenceTxt, tokens, dependencies, d.sourceFilename)
                # Let's gather up the information about the "known" entities in the sentence
                for entityID, entityLocs in sorted(entityIDsToTokenLocs.items()):
                    # Get the entity associated with this ID
                    e = entityIDsToEntities[entityID]
                    sentence.addEntityAnnotation(e, entityLocs)
                d.addSentence(sentence)
                # print(sentence.entityAnnotations)
                # print(d.sentences)
        # print(tagdict)
        # print(depdict)
        
        corpus.parsed = True
