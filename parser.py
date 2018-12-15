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
        Create a Parser object that will use Spacy for parsing. 
        It offers all the same languages that Spacy offers. Check out: https://spacy.io/usage/models. 
        Note that the language model needs to be downloaded first (e.g. python -m spacy download en)
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
        """Parse the document text.
        Then generate sentences based on the parsed text
        """
        if six.PY2 and isinstance(text, str):
            text = unicode(text)
        # TODO: add better sentence segmenter
        # process and use spacy language model to parse the text
        # each parsed tokens in the text contains all infos
        # (word, lemma, dep_head, dep_rel, token_offset, pos, ner)
        text = text.strip().replace("\n", " ").replace("\r", " ")
        text = text.lower()
        parsed = self.nlp(text)
        sentence = None
        # print(parsed)

        # concatenate tokens and yield sentences
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
        Parse the corpus. 
        Each document will be split into sentences which are then tokenized and 
        parsed for their dependency graph. 
        All parsed information is stored within the corpus object.
        :param corpus: Corpus to parse
        :type corpus: kindred.Corpus
        """
        assert isinstance(corpus, kindred.Corpus)
        
        # Document(text, entities, relations, sourceId)
        for d in corpus.documents:
            # create a dict of entities
            entityIDsToEntities = {entity.entityID: entity for entity in d.entities}

            # create an IntervalTree object containing all entities
            # IntervalTree(Entity(start_pos, end_pos, entityID), Entity(...))
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

            # loop over sentences parsed from the document text (spacy parser), 
            # extract tokens' infos from each parsed sentence to create Sentence objects
            # Sentence(text, tokens, dependencies, sourceFilename)
            # and add the objects into the Document object
            for sentence in self._sentencesGenerator(d.text):
                # convert spacy token object into our own token object
                # Token(text, lemma, tags, ner, startPos, endPos) 
                tokens = []
                for t in sentence:
                    ent_type_ = 'O' if t.ent_type_ == '' else t.ent_type_  
                    token = kindred.Token(
                        t.text, t.lemma_, t.tag_, ent_type_, t.idx, t.idx+len(t.text))
                    tokens.append(token)
                    if t.tag_ not in tagdict:
                        tagdict[t.tag_] = len(tagdict)

                # extract sentence text by the first and last tokens positions (charater level)
                sentenceStart = tokens[0].startPos
                sentenceEnd = tokens[-1].endPos
                sentenceTxt = d.text[sentenceStart:sentenceEnd]
                
                # extract dependencies
                dependencies = []
                sentOffset = sentence[0].i  # pos (word level) of first token of a sentence in the whole document
                for t in sentence:
                    depName = t.dep_  # deprel
                    # make the dependency format same as the gcn code
                    if depName == 'ROOT':
                        dep = (0, t.i-sentOffset, depName)  # t.i - sentOffset: token pos - sentOffset
                    else:
                        dep = (t.head.i-sentOffset+1, t.i-sentOffset, depName)
                    dependencies.append(dep)
                # print(tokens)
                # print(entities)
                # print(dependencies)

                # gather tokens inside each entity based on the denotationTree
                # entityIDsToTokenLocs{"entity_id": [token1_pos, token2_pos], ...}
                entityIDsToTokenLocs = defaultdict(list)
                for i, t in enumerate(tokens):
                    entitiesOverlappingWithToken = denotationTree[t.startPos:t.endPos]
                    for interval in entitiesOverlappingWithToken:
                        entityID = interval.data
                        entityIDsToTokenLocs[entityID].append(i)
                # print(entityIDsToTokenLocs)

                # create Sentence object 
                sentence = kindred.Sentence(
                    sentenceTxt, tokens, dependencies, d.sourceFilename)                

                # add entities annotations (entities and their tokens position) in the sentence
                # Annotations([(Entity1, [token1_pos, token2_pos]), (Entity2, [token3_pos, token5_pos])])
                for entityID, entityLocs in sorted(entityIDsToTokenLocs.items()):
                    e = entityIDsToEntities[entityID]  # get the entity associated with this ID
                    # check whether entities are in same position as parsed tokens
                    print(e)
                    for loc in entityLocs:
                        print(tokens[loc])
                    sentence.addEntityAnnotation(e, entityLocs)  #

                # add the Sentence object with contained infos into the Document object
                d.addSentence(sentence)
                # print(sentence.entityAnnotations)
                # print(d.sentences)
        
        corpus.parsed = True
