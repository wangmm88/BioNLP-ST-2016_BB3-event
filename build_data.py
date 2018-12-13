
import json
from collections import defaultdict

import kindred
from parser import Parser

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataBuilder:
    """
    Manages binary classifier(s) for relation classification.

    :param entityCount: Number of entities in each relation (default=2). 
                Passed to the CandidateBuilder (if needed)
    :param acceptedEntityTypes: Tuples of entity types that relations must match. 
                None will match allow relations of any entity types.
                Passed to the CandidateBuilder (if needed)
    """

    def __init__(self, entityCount=2, acceptedEntityTypes=None):
      
        assert isinstance(entityCount, int)
        assert acceptedEntityTypes is None or isinstance(acceptedEntityTypes, list)

        self.entityCount = entityCount
        self.acceptedEntityTypes = acceptedEntityTypes

    def build(self, corpus):
        """
        Trains the classifier using this corpus. All relations in the corpus will be used for training.

        :param corpus: Corpus to use for training
        :type corpus: kindred.Corpus
        """
        assert isinstance(corpus, kindred.Corpus)
        # TODO: break this method into multiple methods following deep learning format
        logger.info('start_parsing')
        if not corpus.parsed:
            parser = Parser()
            parser.parse(corpus)

        self.candidateBuilder = kindred.CandidateBuilder(
            entityCount=self.entityCount, acceptedEntityTypes=self.acceptedEntityTypes)
        candidateRelations = self.candidateBuilder.build(corpus)
        # logger.info('%s\n', candidateRelations[10:15])
        
        if len(candidateRelations) == 0:
            raise RuntimeError("No candidate relations found in corpus for training.")

        data = []
        for cr in candidateRelations:
            cr_dict = defaultdict()
            cr_dict['id'] = str(cr.sentence.sourceFilename)
            cr_dict['subj_entity'] = str(cr.entities[0])
            cr_dict['obj_entity'] = str(cr.entities[1])
            cr_dict['relation'] = 0 if len(cr.knownTypesAndArgNames) == 0 else 1

            token = []
            stanford_pos = []
            for t in cr.sentence.tokens:
                token.append(t.word)
                stanford_pos.append(t.partofspeech)
            cr_dict['token'] = token
            cr_dict['stanford_pos'] = stanford_pos
            # cr_dict['stanford_ner'] = stanford_ner
            
            stanford_head = []
            stanford_deprel = []
            for dep in cr.sentence.dependencies:
                stanford_head.append(str(dep[0]))
                stanford_deprel.append(dep[-1])
            cr_dict['stanford_head'] = stanford_head
            cr_dict['stanford_deprel'] = stanford_deprel
            
            subj_start, subj_end = getPos(cr.entities[0], cr.sentence.entityAnnotations)
            obj_start, obj_end = getPos(cr.entities[1], cr.sentence.entityAnnotations)
            cr_dict['subj_start'] = subj_start
            cr_dict['subj_end'] = subj_end
            cr_dict['obj_start'] = obj_start
            cr_dict['obj_end'] = obj_end
            
            cr_dict['subj_type'] = cr.entities[0].entityType
            cr_dict['obj_type'] = cr.entities[1].entityType
            
            stanford_ner = ['O' for _ in range(len(token))]
            for idx in range(subj_start, subj_end+1):
                stanford_ner[idx] = cr_dict['subj_type']
            for idx in range(obj_start, obj_end+1):
                stanford_ner[idx] = cr_dict['obj_type']
            cr_dict['stanford_ner'] = stanford_ner
            
            # for entityAnno in cr.sentence.entityAnnotations:
            #     entity, entityLocs = entityAnno
            #     if entity.entityType in ['Bacteria', 'Habitat', 'Geographical']:
            #         print(entity)
            #         print(entity.entityType)
            #         for idx in entityLocs:
            #             stanford_ner[idx] = entity.entityType
    
            data.append(cr_dict)

        print('Data length: %d' %len(data))
        return data

def getPos(entity, entityAnnotations):
    for entityAnno in entityAnnotations:
        if entity == entityAnno[0]:
            return entityAnno[1][0], entityAnno[1][-1]

if __name__=="__main__":
    dataset_names = ['train', 'dev', 'test']
    # dataset_names = ['test']
    builder = DataBuilder(entityCount=2, acceptedEntityTypes=[('Bacteria', 'Habitat'), ('Bacteria', 'Geographical')])
    dirname = 'dataset/bb3/BioNLP-ST-2016_BB-event_'
    for dn in dataset_names:
        print(dn)
        corpus = kindred.loadDir(
            dataFormat='standoff', directory=dirname+dn, ignoreEntities=[])
        data = builder.build(corpus)
        with open('dataset/bb3/' + dn + '.json', 'w') as fp:
            json.dump(data, fp, indent=2)
