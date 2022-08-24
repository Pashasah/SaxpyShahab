# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:32:45 2022

@author: sp7012
"""

from saxpy import sax
import numpy as np

class TestSAX(object):
    def setUp(self):
        # All tests will be run with 6 letter words
        # and 5 letter alphabet
        self.sax = sax(6, 5, 1e-6)

    def test_to_letter_rep(self):
        arr = [7,1,4,4,4,4]
        (letters, indices) = self.sax.to_letter_rep(arr)
        assert letters == 'eacccc'

    def test_to_letter_rep_missing(self):
        arr = [7,1,4,4,np.nan,4]
        (letters, indices) = self.sax.to_letter_rep(arr)
        assert letters == 'eacc-c'

    def test_long_to_letter_rep(self):
        long_arr = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,6,6,6,6,10,100]
        (letters, indices) = self.sax.to_letter_rep(long_arr)
        assert letters == 'bbbbce'

    def test_long_to_letter_rep_missing(self):
        long_arr = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,np.nan,1,1,6,6,6,6,10,100]
        (letters, indices) = self.sax.to_letter_rep(long_arr)
        assert letters == 'bbb-ce'

    def test_compare_strings(self):
        base_string = 'aaabbc'
        similar_string = 'aabbbc'
        dissimilar_string = 'ccddbc'
        similar_score = self.sax.compare_strings(base_string, similar_string)
        dissimilar_score = self.sax.compare_strings(base_string, dissimilar_string)
        assert similar_score < dissimilar_score

    def test_compare_strings_missing(self):
        assert self.sax.compare_strings('a-b-c-', 'b-c-d-') == 0

    def test_normalize_missing(self):
        # two arrays which should normalize to the same result
        # except one should contain a nan value in place of the input nan value
        incomplete_arr_res = self.sax.normalize([1,0,0,0,0,1,np.nan])
        complete_arr_res = self.sax.normalize([1,0,0,0,0,1])
        assert np.array_equal(incomplete_arr_res[:-1], complete_arr_res)
        assert np.isnan(incomplete_arr_res[-1])
    def test_normalize_under_epsilon(self):
        array_under_epsilon = self.sax.normalize([1e-7, 2e-7, 1.5e-7])
        assert np.array_equal(array_under_epsilon, [0,0,0])
        
        
        
        
        
        """Testing SAX implementation."""
import numpy as np
from saxpy.sax import sax_by_chunking


def test_chunking():
    """Test SAX by chunking."""
    dat1 = np.array([2.02, 2.33, 2.99, 6, 5.85,
                     3.85, 4.85, 3.85, 2.22, 1.45, 1.34])

    dats1_9_7 = "bcggfddba"
    dats1_10_11 = "bcjkjheebb"
    dats1_14_10 = "bbdijjigfeecbb"

    assert dats1_9_7 == sax_by_chunking(dat1, 9, 7)
    assert dats1_10_11 == sax_by_chunking(dat1, 10, 11)
    assert dats1_14_10 == sax_by_chunking(dat1, 14, 10)

    dat2 = np.array([0.5, 1.29, 2.58, 3.83,2.63, 3.29, 1.58, 3.83, 2.63, 8.44, 1.25,
                    8.75, 8.83, 3.25, 0.75, 8.72,4.29, 2.83,2.63, 2.29])

    dats2_9_7 = "accdefgda"
    dats2_10_11 = "bcefgijkcb"
    dats2_14_10 = "abdeeffhijjfbb"

    assert dats2_9_7 == sax_by_chunking(dat2, 8, 7)
    assert dats2_10_11 == sax_by_chunking(dat2, 10, 11)
    assert dats2_14_10 == sax_by_chunking(dat2, 12, 10)
    
    
    
    
from saxpy.alphabet import cuts_for_asize
cuts_for_asize(3)
import numpy as np
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
ts_to_string(znorm(np.array([-2, 0, 2, 0, -1])), cuts_for_asize(3))