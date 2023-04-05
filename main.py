import math
from collections import defaultdict
from typing import List, Set, Tuple, Dict, Iterable

from utils import Message, tokenize


class NaiveBayesClassifier:
	"""Perform Multinomial Naive Bayes.
	"""

	def __init__(self, k: float = 0.5) -> None:
		self.k = k                                                     # smoothing factor

		self.tokens: Set[str] = set()                                  # vocabulary
		self.token_spam_counts: Dict[str, int] = defaultdict(int)      # vocabulary and counts for spam messages
		self.token_ham_counts: Dict[str, int] = defaultdict(int)       # vocabulary and counts for ham (non-spam) messages
		self.spam_messages = self.ham_messages = 0                     # number of spam and ham messages

	def train(self, messages: Iterable[Message]) -> None:
		for message in messages:
			# increment message counts
			if message.is_spam:
				self.spam_messages += 1
			else:
				self.ham_messages += 1

			# increment word counts
			for token in tokenize(message.text):             # get the text attribute of message
				self.tokens.add(token)                       # add the new token to the vocabulary set

				if message.is_spam:
					self.token_spam_counts[token] += 1
				else:
					self.token_ham_counts[token] += 1

	def _probs(self, token: str) -> Tuple[float, float]:
		"""Private helper function to return P(token|spam) and P(token|ham)
		for a specific token.
		"""

		spam = self.token_spam_counts[token]      # extract the counts for specific token in spam dict
		ham = self.token_ham_counts[token]        # extract the counts for specific token in ham dict

		p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
		p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

		return p_token_spam, p_token_ham

	def predict(self, text: str) -> float:
		"""Classify a new text message as spam or ham.

		NOTE:
		- We assume the prior probabilities P(spam) and P(ham) are equal,
		so we don't include it in the computation of Bayes rule.
		- Rather than multiplying together lots of small probabilities,
		we'll sum up the log probabilities to avoid underflow.
		"""

		text_tokens = tokenize(text)  # tokenize the new text to be predicted
		log_prob_if_spam = log_prob_if_ham = 0.0

		# iterate through each word in our vocabulary
		for token in self.tokens:
			prob_if_spam, prob_if_ham = self._probs(token)  # compute P(token|spam) and P(token|ham)

			# if *token* appears in the message, add the log probability of seeing it
			if token in text_tokens:
				log_prob_if_spam += math.log(prob_if_spam)
				log_prob_if_ham += math.log(prob_if_ham)
			# otherwise, add the log probability of _not_ seeing it, which is log(1 - probability of seeing it)
			else:
				log_prob_if_spam += math.log(1.0 - prob_if_spam)
				log_prob_if_ham += math.log(1.0 - prob_if_ham)

		# convert log(probability) to normal probability
		prob_if_spam = math.exp(log_prob_if_spam)
		prob_if_ham = math.exp(log_prob_if_ham)

		return prob_if_spam / (prob_if_spam + prob_if_ham)
