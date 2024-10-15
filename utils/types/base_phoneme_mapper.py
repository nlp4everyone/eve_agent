from typing import List, Dict
import regex as re

class BasePhonemeMapper:
    def __init__(self):
        """Mapping class from phoneme to viseme"""

    @staticmethod
    def _normalize_ipa(words: List[str]) -> List[str]:
        """Normalize some type of word. For example: j-u: to ju:"""
        for (i, word) in enumerate(words):
            words[i] = re.sub(r"j-uː", "juː", words[i])
        return words

    def _mapping_logic(self,
                      list_phoneme :List[str],
                      mapping_dict :Dict[str,str]):
        # Mapping
        map = []
        for phoneme in list_phoneme:
            for key in dict(mapping_dict).keys():
                if phoneme in mapping_dict[key]:
                    # When no syllable
                    if len(map) == 0:
                        map.append(key)
                    # Not overlap
                    elif key != map[-1]:
                        map.append(key)
                    break
        return map
