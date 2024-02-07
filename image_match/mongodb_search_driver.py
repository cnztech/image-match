from .signature_database_base import SignatureDatabaseBase
from .signature_database_base import normalized_distance
from datetime import datetime
import numpy as np
from collections import deque


class SignatureMongoSearch(SignatureDatabaseBase):
    """Mongodb Atlas Search driver for image-match

    """

    SIMPLE_WORD_PREFIX = 'simple_word_'
    SHORT_SIMPLE_WORD_PREFIX = 'sw_'

    def __init__(self, collection, index_name, size=100, *args, **kwargs):
        """Extra setup for Atlas Search

        Args:
            collection: the mongodb collection
            index_name: name of the search index in the collection
            size: maximum number of results to return
            *args (Optional): Variable length argument list to pass to base constructor
            **kwargs (Optional): Arbitrary keyword arguments to pass to base constructor
        """
        self.collection = collection
        self.index_name = index_name
        self.size = size
        super(SignatureMongoSearch, self).__init__(*args, **kwargs)

    def search_single_record(self, rec, pre_filter=None):
        path = rec.pop('path')
        signature = rec.pop('signature')
        if 'metadata' in rec:
            rec.pop('metadata')

        rec = self._stringify_simple_words(rec)

        query = {
            'should': [{'text': {'path': word, 'query': str(rec[word])}} for word in rec]
        }

        if pre_filter is not None:
            query['must'] = pre_filter

        res = list(self.collection.aggregate([
            {'$search': {
                'index': self.index_name,
                'compound': query
            }},
            {'$limit': self.size},
            {'$project': {'_id': 1, 'metadata': 1, 'path': 1, 'signature': 1}},
        ]))

        sigs = np.array([x['signature'] for x in res])

        if sigs.size == 0:
            return []

        dists = normalized_distance(sigs, np.array(signature))

        formatted_res = [{'id': str(x['_id']),
                          'metadata': x.get('metadata'),
                          'path': x.get('path')}
                         for x in res]

        for i, row in enumerate(formatted_res):
            row['dist'] = dists[i]
        formatted_res = filter(lambda y: y['dist'] < self.distance_cutoff, formatted_res)

        return formatted_res

    def insert_single_record(self, rec, refresh_after=False):
        rec = self._stringify_simple_words(rec)
        rec['timestamp'] = datetime.now()
        self.collection.update_one({'path': rec['path']}, {'$set': rec}, upsert=True)

        # if the collection has no indexes (except possibly '_id'), build them
        if len(self.collection.index_information()) <= 1:
            self.index_collection()

    def index_collection(self):
        """Index a collection on words.

        """
        self.collection.create_index({'path': 1}, unique=True)
    
    def delete_image(self, path):
        """Delete an image from the database."""
        self.collection.delete_one({'path': path})

    def is_image_existing(self, path):
        """Check if an image is already in the database."""
        return True if self.collection.find_one({'path': path}) else False

    def _stringify_simple_words(self, rec):
        result = {}
        for key, value in rec.items():
            if not key.startswith(self.SIMPLE_WORD_PREFIX):
                result[key] = value
                continue
            result[f'{self.SHORT_SIMPLE_WORD_PREFIX}{key[len(self.SIMPLE_WORD_PREFIX):]}'] = str(value)
        return result
    
    def _restore_to_simple_words(self, rec):
        result = {}
        for key, value in rec.items():
            if not key.startswith(self.SHORT_SIMPLE_WORD_PREFIX):
                result[key] = value
                continue
            result[f'{self.SIMPLE_WORD_PREFIX}{key[len(self.SHORT_SIMPLE_WORD_PREFIX):]}'] = int(value)
        return result