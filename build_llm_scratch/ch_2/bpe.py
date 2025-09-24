import string
from collections import Counter
from typing import Optional
from collections import defaultdict

class TextPreprocessor:
    def __init__(self, end_of_word_token: str = "</w>", lowercase: bool = True):
        self.end_of_word_token = end_of_word_token
        self.lowercase = lowercase
    
    def normalize_text(self, text: str) -> str:
        # Clean and standardize text (lowercase, remove special chars, etc.)
        table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        clean_text = text.translate(table)
        return clean_text
    
    def tokenize_to_words(self, text: str) -> list[str]:
        # Split text into individual words
        norm = self.normalize_text(text)
        if self.lowercase:
          norm = norm.lower()
        return norm.split()
    
    def words_to_char_tuples(self, words: list[str]) -> list[tuple[str, ...]]:
        # Convert words to character sequences with end-of-word markers
        return [tuple(list(word) + [self.end_of_word_token]) for word in words]
    
    def build_word_frequencies(self, corpus: list[str]) -> dict[tuple[str, ...], int]:
        # Count how often each word appears in corpus
        words = []
        for text in corpus:
          words.extend(self.tokenize_to_words(text))     

        char_tuples = self.words_to_char_tuples(words)       
        return Counter(char_tuples)
    
    def preprocess_corpus(self, corpus: list[str]) -> dict[tuple[str, ...], int]:
        # Main entry point: load corpus and convert to training format
        return self.build_word_frequencies(corpus)


class VocabularyManager:
    def __init__(self, end_of_word_token: str = "</w>"):
        self.end_of_word_token = end_of_word_token
        self.base_vocab = {}
    
    def initialize_base_vocab(self, word_frequencies: dict[tuple[str, ...], int]) -> None:
        # Create initial vocabulary from unique characters in corpus
        unique_symbols = set()
        for word in word_frequencies:
          unique_symbols.update(word)

        self.base_vocab = {sym: idx for idx, sym in enumerate(sorted(unique_symbols))} 
    
    def add_merge_token(self, token: str) -> int:
        # Add new merged token to vocabulary and assign ID
        token_id = len(self.base_vocab)
        self.base_vocab[token] = token_id
        return token_id
    
    def get_token_id(self, token: str) -> Optional[int]:
        # Look up token ID by token string
        return self.base_vocab.get(token)
    
    def get_token_by_id(self, token_id: int) -> Optional[str]:
        # Look up token string by ID
        for token, id_ in self.base_vocab.items():
          if id_ == token_id:
            return token
        return None
    
    def get_vocab(self) -> dict[str, int]:
        # Return complete token→ID mapping
        return self.base_vocab
    
    def get_reverse_vocab(self) -> dict[int, str]:
        # Return complete ID→token mapping
        return {v: k for k, v in self.base_vocab.items()}
    
    def vocab_size(self) -> int:
        return len(self.base_vocab)
    
    def contains_token(self, token: str) -> bool:
        return token in self.base_vocab


class PairFrequencyAnalyzer:
    def __init__(self, tie_breaking: str = "lexicographic"):
        self.tie_breaking = tie_breaking
    
    def count_all_pairs(self, word_frequencies: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
        # Count frequency of every adjacent token pair across corpus
        pair_counts = defaultdict(int)
        for word, freq in word_frequencies.items():
          pairs = self.get_pairs_from_word(word)
          for pair in pairs:
            pair_counts[pair] += freq
        return dict(pair_counts)
    
    def get_pairs_from_word(self, word: tuple[str, ...]) -> List[tuple[str, str]]:
        # Extract all adjacent pairs from a single word
        return [(word[i], word[i+1]) for i in range(len(word)-1)]
    
    def find_most_frequent_pair(self, pair_counts: dict[tuple[str, str], int]) -> Optional[Tuple[str, str]]:
        # Identify the pair to merge next (handles ties)
        if not pair_counts:
          return None
        
        max_count = max(pair_counts.values())
        candidates = [pair for pair, count in pair_counts.items() if count == max_count]

        if len(candidates) == 1:
          return candidates[0]
        else:
          return self.resolve_tie(candidates)
    
    def resolve_tie(self, tied_pairs: list[tuple[str, str]]) -> tuple[str, str]:
        # Apply tie-breaking strategy (lexicographic/random/etc.)
        if self.tie_breaking == 'lexicographic':
          return sorted(tied_pairs)[0]
        else:
          import random
          return random.choice(tied_pairs)
    
    def update_pair_counts_after_merge(self, word_frequencies: dict[tuple[str, ...], int], 
                                     merged_pair: tuple[str, str]) -> dict[tuple[str, str], int]:
        # Efficiently recalculate pair frequencies after a merge
        new_word_freqs = {}
        for word, freq in word_frequencies.items():
          new_word = []
          i = 0
          while i < len(word):
            # merge pair
            if i < len(word)-1 and (word[i], word[i+1]) == merged_pair:
              new_word.append(word[i] + word[i+1])
              i+=2
            else:
              new_word.append(word[i])
              i+=1
          new_word_freqs[tuple(new_word)] = freq
        return self.count_all_pairs(new_word_freqs)
    
class MergeEngine:
    def __init__(self):
        self.merges = [] # list of tuples: ((A, B), 'AB')
    
    def create_merge_token(self, pair: Tuple[str, str]) -> str:
        # Generate new token name from pair (e.g., ('e','s') → 'es')
        return pair[0] + pair[1]
    
    def apply_merge_to_word(self, word: Tuple[str, ...], pair: Tuple[str, str], 
                           new_token: str) -> Tuple[str, ...]:
        # Replace pair occurrences in single word
        new_word = []
        i = 0
        while i < len(word):
          if i < len(word) - 1 and (word[i], word[i+1]) == pair:
            new_word.append(new_token)
            i += 2
          else:
            new_word.append(word[i])
            i += 1
        return tuple(new_word)
    
    def apply_merge_to_corpus(self, word_frequencies: Dict[Tuple[str, ...], int], 
                             pair: Tuple[str, str], new_token: str) -> Dict[Tuple[str, ...], int]:
        # Apply merge across entire corpus
        new_word_freqs = {}
        for word, freq in word_frequencies.items():
          merged_word = self.apply_merge_to_word(word, pair, new_token)
          new_word_freqs[merged_word] = freq
        return new_word_freqs
    
    def get_merge_rules(self) -> List[Tuple[Tuple[str, str], str]]:
        # Return ordered list of all merges performed
        return self.merges
    
    def add_merge_rule(self, pair: Tuple[str, str], new_token: str) -> None:
        # Record a merge operation for later use
        return self.merges.append((pair, new_token))
    
class BPETrainer:
    def __init__(self, target_vocab_size: int, preprocessor: TextPreprocessor, 
                 vocab_manager: VocabularyManager, pair_analyzer: PairFrequencyAnalyzer, 
                 merge_engine: MergeEngine):
      self.target_vocab_size = target_vocab_size
      self.preprocessor = preprocessor
      self.vocab_manager = vocab_manager
      self.pair_analyzer = pair_analyzer
      self.merge_engine = merge_engine        
    
    def train_from_corpus(self, corpus: List[str], verbose: bool = False) -> Tuple[Dict[str, int], List[Tuple[Tuple[str, str], str]]]:
        # Main training entry point - orchestrates entire process
        word_freqs = self.preprocessor.preprocess_corpus(corpus)
        self.vocab_manager.initialize_base_vocab(word_freqs)
        merge_rules = self.perform_merge_iterations(word_freqs, verbose=verbose)
        return self.vocab_manager.get_vocab(), merge_rules
    
    def perform_merge_iterations(self, word_frequencies: Dict[Tuple[str, ...], int], 
                               verbose: bool = False) -> List[Tuple[Tuple[str, str], str]]:
        # Execute iterative merging until vocabulary target reached
        merge_rules = []
        while self.should_continue_merging(self.vocab_manager.vocab_size(), self.pair_analyzer.count_all_pairs(word_frequencies)):
          pair_counts = self.pair_analyzer.counts_all_pair(word_frequencies)
          best_pair = self.pair_analyzer.find_most_frequent_pairs(pair_counts)
          if not best_pair:
            break
          new_token = self.merge_engine.create_merge_token(best_pair)
          self.vocab_manager.add_merge_token(new_token)
          self.merge_engine.add_merge_rule(best_pair, new_token)
          word_frequencies = self.merge_engine.apply_merge_to_corpus(word_frequencies, best_pair, new_token)
          if verbose:
            self.log_merge_progress(iteration=len(merge_rules)+1, pair=best_pair, 
                                    frequency=pair_counts[best_pair], 
                                    vocab_size=self.vocab_manager.vocab_size())
          merge_rules.append((best_pair, new_token))
        return merge_rules
    
    def should_continue_merging(self, current_vocab_size: int, pair_counts: Dict[Tuple[str, str], int]) -> bool:
        # Determine if training should continue
        if current_vocab_size >= self.target_vocab_size:
          return False
        if not pair_counts:
          return False
        return True
    
    def log_merge_progress(self, iteration: int, pair: Tuple[str, str], frequency: int, 
                          vocab_size: int) -> None:
        # Display training progress information
        print(f"Iteration {iteration}: Merged pair {pair} (freq={frequency}), vocab_size={vocab_size}")

class BPEEncoder:
    def __init__(self, vocab: Dict[str, int], merge_rules: List[Tuple[Tuple[str, str], str]], 
                 preprocessor: TextPreprocessor):
        self.vocab = vocab
        self.merge_rules = merge_rules
        self.preprocessor = preprocessor
    
    def encode_text(self, text: str) -> List[int]:
        # Main encoding entry point - text to token IDs
        words = self.preprocessor.tokenize_to_words(text)
        token_ids = []
        for word in words:
          token_ids.extend(self.encode_word(word))
        return token_ids
    
    def encode_word(self, word: str) -> List[int]:
        # Encode single word using BPE rules
        word_chars = tuple(list(word) + [self.preprocessor.end_of_word_token])
        tokens = self.apply_bpe_to_word(word_chars)
        tokens = self.handle_unknown_tokens(tokens)
        return self.tokens_to_ids(tokens)
    
    def apply_bpe_to_word(self, word_chars: Tuple[str, ...]) -> List[str]:
        # Apply merge rules to segment word into subwords
        tokens = list(word_chars)
        return self.apply_merge_rules(tokens)
    
    def apply_merge_rules(self, tokens: List[str]) -> List[str]:
        # Apply learned merges in training order
        for pair, merged_token in self.merge_rules:
          i = 0
          while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) == pair:
              tokens[i] = merged_token
              del tokens[i+1]
            else:
              i +=1
    
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        # Convert token strings to vocabulary IDs
        return [self.vocab[token] for token in tokens if token in self.vocab]
    
    def handle_unknown_tokens(self, tokens: List[str]) -> List[str]:
        # Handle tokens not in vocabulary (fallback to characters)
        result = []
        for t in tokens:
          if t in self.vocab:
            result.append(t)
          else:
            result.extend(list(t))
        return result
    
class BPEDecoder:
    def __init__(self, id_to_token: Dict[int, str], end_of_word_token: str = "</w>"):
        self.id_to_token = id_to_token
        self.end_of_word_token = end_of_word_token
    
    def decode_ids(self, token_ids: List[int]) -> str:
        # Main decoding entry point - token IDs to text
        tokens = self.ids_to_tokens(token_ids)
        words = self.reconstruct_words(tokens)
        return self.join_words(words)
    
    def ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        # Convert IDs to token strings
        return [self.id_to_token[i] for i in token_ids if i in self.id_to_token]
    
    def merge_subword_tokens(self, tokens: List[str]) -> str:
        # Join subword tokens back into readable text
        return "".join(token.replace(self.end_of_word_token, "") for token in tokens)
    
    def reconstruct_words(self, tokens: List[str]) -> List[str]:
        # Group subword tokens back into complete words
        words = []
        current_words = []

        for token in tokens:
          if token.endswith(self.end_of_word_token):
            token_clean = token.replace(self.end_of_word_token, "")
            current_word.append(token_clean)
            words.append("".join(current_word))
          else:
            current_word.append(token)
        
        if current_word:
          words.append("".join(current_word))
        return words
    
    def join_words(self, words: List[str]) -> str:
        # Join words with appropriate spacing
        return " ".join(words)

import json
from typing import Dict, List, Tuple, Any

class ModelSerializer:
    @staticmethod
    def save_model(vocab: Dict[str, int], merge_rules: List[Tuple[Tuple[str, str], str]], 
                   config: Dict[str, Any], filepath: str) -> None:
        merge_rules_serializable = [[list(pair), merged] for pair, merged in merge_rules]
        data = {
            "vocab": vocab,
            "merge_rules": merge_rules_serializable,
            "config": config
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_model(filepath: str) -> Tuple[Dict[str, int], List[Tuple[Tuple[str, str], str]], Dict[str, Any]]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        merge_rules = [(tuple(pair), merged) for pair, merged in data["merge_rules"]]
        return data["vocab"], merge_rules, data["config"]

    @staticmethod
    def save_vocab_only(vocab: Dict[str, int], filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_vocab_only(filepath: str) -> Dict[str, int]:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def export_merge_rules(merge_rules: List[Tuple[Tuple[str, str], str]], filepath: str) -> None:
        merge_rules_serializable = [[list(pair), merged] for pair, merged in merge_rules]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(merge_rules_serializable, f, ensure_ascii=False, indent=2)

    @staticmethod
    def import_merge_rules(filepath: str) -> List[Tuple[Tuple[str, str], str]]:
        with open(filepath, "r", encoding="utf-8") as f:
            merge_rules_serializable = json.load(f)
        return [(tuple(pair), merged) for pair, merged in merge_rules_serializable]
    
class BPETokenizer:
    def __init__(self, vocab_size: int = 10000, end_of_word_token: str = "</w>", 
                 tie_breaking: str = "lexicographic"):
      self.vocab_size = vocab_size
      self.end_of_word_token = end_of_word_token
      self.tie_breaking = tie_breaking

      # Components
      self.preprocessor = TextPreprocessor(end_of_word_token=self.end_of_word_token)
      self.vocab_manager = VocabularyManager(end_of_word_token=self.end_of_word_token)
      self.pair_analyzer = PairFrequencyAnalyzer(tie_breaking=self.tie_breaking)
      self.merge_engine = MergeEngine()
      self.trainer = BPETrainer(
          target_vocab_size=self.vocab_size,
          preprocessor=self.preprocessor,
          vocab_manager=self.vocab_manager,
          pair_analyzer=self.pair_analyzer,
          merge_engine=self.merge_engine
      )

      # Encoder/Decoder placeholders
      self.encoder = None
      self.decoder = None
      self.config = {
          "vocab_size": self.vocab_size,
          "end_of_word_token": self.end_of_word_token,
          "tie_breaking": self.tie_breaking
      }
    
    def train(self, corpus: List[str], verbose: bool = False) -> None:
        # Load and process text corpus, train BPE model
        vocab, merge_rules = self.trainer.train_from_corpus(corpus, verbose=verbose)
        self.encoder = BPEEncoder(vocab=vocab, merge_rules=merge_rules, preprocessor=self.preprocessor)
        id_to_token = {i: tok for tok, i in vocab.items()}
        self.decoder = BPEDecoder(id_to_token=id_to_token, end_of_word_token=self.end_of_word_token)
        self.merge_rules = merge_rules

    
    def encode(self, text: str) -> List[int]:
        # Convert input text to token IDs using trained model
        if self.encoder is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.encoder.encode_text(text)
    
    def decode(self, token_ids: List[int]) -> str:
        # Convert token IDs back to readable text
        if self.decoder is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.decoder.decode_ids(token_ids)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        # Batch encoding for efficiency
        return [self.encode(t) for t in texts]
    
    def decode_batch(self, token_ids_batch: List[List[int]]) -> List[str]:
        # Batch decoding for efficiency
        return [self.decode(ids) for ids in token_ids_batch]
    
    def get_vocab_size(self) -> int:
        return self.vocab_manager.vocab_size() if self.vocab_manager else 0
    
    def save(self, filepath: str) -> None:
        # Persist trained model to disk
        if self.encoder is None or self.decoder is None:
            raise ValueError("Tokenizer not trained yet")
        ModelSerializer.save_model(
            vocab=self.encoder.vocab,
            merge_rules=self.merge_rules,
            config=self.config,
            filepath=filepath
        )
    
    def load(self, filepath: str) -> None:
        # Load pre-trained model from disk
        vocab, merge_rules, config = ModelSerializer.load_model(filepath)
        self.config = config
        self.encoder = BPEEncoder(vocab=vocab, merge_rules=merge_rules, preprocessor=self.preprocessor)
        id_to_token = {i: tok for tok, i in vocab.items()}
        self.decoder = BPEDecoder(id_to_token=id_to_token, end_of_word_token=self.end_of_word_token)
        self.merge_rules = merge_rules