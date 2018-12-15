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
        
        # parse each document in the corpus
        for d in corpus.documents:
            # create a dict of entity
            entityIDsToEntities = {entity.entityID: entity for entity in d.entities}
            # create interval tree containing all entities
            # Format: IntervalTree(Entity(start_pos, end_pos, entityID), Entity(...))
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

            # generate sentence from document text, extract tokens' infos
            for sentence in self._sentencesGenerator(d.text):
                # convert spacy token object into our own token object
                # Format: Token(text, lemma, tags, ner, startPos, endPos) 
                tokens = []
                for t in sentence:
                    ent_type_ = 'O' if t.ent_type_ == '' else t.ent_type_  
                    token = kindred.Token(
                        t.text, t.lemma_, t.tag_, ent_type_, t.idx, t.idx+len(t.text))
                    tokens.append(token)
                    if t.tag_ not in tagdict:
                        tagdict[t.tag_] = len(tagdict)

                # extract sentence text by the first and last tokens position
                sentenceStart = tokens[0].startPos
                sentenceEnd = tokens[-1].endPos
                sentenceTxt = d.text[sentenceStart:sentenceEnd]
                indexOffset = sentence[0].i  # sentence offset in the document
                
                # extract dependencies
                dependencies = []
                for t in sentence:
                    depName = t.dep_  # deprel
                    # make the dependency format same as the gcn code
                    if depName == 'ROOT':
                        dep = (0, t.i-indexOffset, depName)
                    else:
                        dep = (t.head.i-indexOffset+1, t.i-indexOffset, depName)
                    dependencies.append(dep)
                # print(tokens)
                # print(entities)
                # print(dependencies)

                # gather tokens inside each entity
                # Format {"entity_id": [token1, token2], ...}
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

                # gather the entities and their tokens position in a sentence
                # Annotation Format [(Entity1, [token1, token2]), (Entity2, [token3, token5])]
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
