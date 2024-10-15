from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from typing import List, Union,Literal
from ..utils.types import BasePhonemeMapper
import json, os
# pyphen = pyphen.Pyphen(lang="en")

class PhonemizerMapper(BasePhonemeMapper):
    def __init__(self,
                 separator :str = "-",
                 lang :Union[Literal["en-gb","en-us"],str] = "en-gb"):
        super().__init__()
        """Mapping class with supported by Phonemizer"""
        self._lang = lang
        # Define params
        self.__phonemizer_backend = EspeakBackend(self._lang) # Default backend with EN-UK (For US, type en-us)
        # Default separator
        self.__separator = Separator(phone = separator, word=' ')

    def _phonemize(self,
                  words: Union[str,List[str]],
                  strip: bool = True,
                  n_job :int = 1) -> List[str]:
        """
        Return phonemized format of list words
        :param words: Word (str) or list of word
        :param strip: Whether string could be striped or not
        :param n_job: Number of multithread job
        :return: List of phonemized words
        """

        # Convert string to list
        if isinstance(words,str): words = [words]
        # In case empty list
        if len(words) == 0: raise Exception("Words is empty")

        # Return phonemized format
        return self.__phonemizer_backend.phonemize(text = words,
                                                  separator = self.__separator,
                                                  strip = strip,
                                                  njobs = n_job)

    @property
    def mapping_dict(self, mapping_dir: str = "mapping_rules"):
        """
        Property contains an dictionary of mapping with language
        :param mapping_dir: Path to dictionary
        :return:
        """
        # Check path valid
        if not os.path.exists(mapping_dir):
            raise FileNotFoundError

        file_path = None
        # Define path
        if self._lang == "en-us":
            file_path = os.path.join(mapping_dir, "us_rule.json")
        elif self._lang == "en-gb":
            file_path = os.path.join(mapping_dir, "uk_rule.json")

        # Check path existed!
        if not os.path.exists(file_path):
            raise Exception(f"{file_path} is not existed!")

        # Load mapping rule
        with open(file_path, 'r') as file:
            return json.load(file)

    def word_to_viseme(self,words :Union[List[str],str]):
        """
        Function for mapping from word to viseme
        :param words: Word or list or word you needed for converting to viseme
        :return:
        """
        # Convert str to array
        if isinstance(words,str):
            words = [words]

        # Get phonemized format
        phonemized_words = self._phonemize(words = words)
        # Normalized words
        normalized_words = self._normalize_ipa(words=phonemized_words)
        # Mapping phoneme to viseme
        return [self._phoneme_to_viseme(word) for word in normalized_words]

    def _phoneme_to_viseme(self, word_phoneme :str):
        """
        Return a list of viseme from phonemized words
        :param word_phoneme: String word after phonemized
        :return:
        """
        # Convert to list of phoneme
        list_phoneme = word_phoneme.split("-")

        # Get mapping dict
        mapping_dict = self.mapping_dict
        # Define logic
        return self._mapping_logic(list_phoneme = list_phoneme,mapping_dict = mapping_dict)