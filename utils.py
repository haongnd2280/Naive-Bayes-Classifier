import re
from typing import Set, NamedTuple


# define a type for our training data
class Message(NamedTuple):
    """Input for the class
    """

    text: str
    is_spam: bool


def tokenize(text: str) -> Set[str]:
    """Split a raw text message into a set of distinct words.
    """

    text = text.lower()                               # convert to lowercase
    all_words = re.findall("[a-z0-9']+", text)        # extract words (numbers, letters, apostrophes)
    return set(all_words)                             # remove duplicates
