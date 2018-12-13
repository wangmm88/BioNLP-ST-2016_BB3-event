"""
Importer for BioNLP Shared Task data
"""

import os
import tempfile
import sys
import shutil
import six

import kindred

# Load the task information into this data structure
taskOptions = {}
bionlpstFile = os.path.join(os.path.dirname(__file__), 'bionlpst_files.txt')
with open(bionlpstFile, 'r') as f:
    for line in f:
        if line.strip() != '':
            taskName, url, expectedFile, expectedSHA256 = line.strip().split('\t')
            taskOptions[taskName] = (url, expectedFile, expectedSHA256)


def listTasks():
    """
    List the names of the BioNLP Shared Task datasets that can be loaded. These values can be passed to the kindred.bionlpst.load function as the taskName argument

    :return: List of valid taskNames
    :rtype: str
    """

    return sorted(list(taskOptions.keys()))


def load(dirname, ignoreEntities=[]):
    corpus = kindred.loadDir(dataFormat='standoff',
                             directory=dirname, ignoreEntities=ignoreEntities)

    shutil.rmtree(tempDir)

    return corpus
