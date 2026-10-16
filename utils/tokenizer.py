from collections import Counter


# =========================================================
# Special Tokens
# =========================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"


# =========================================================
# Simple Tokenizer
# =========================================================

class SimpleTokenizer:

    def __init__(self, vocab_size=1000):

        self.vocab_size = vocab_size

        # Special Tokens
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.cls_token = CLS_TOKEN
        self.sep_token = SEP_TOKEN

        # Special Token IDs
        self.pad_id = 0
        self.unk_id = 1
        self.cls_id = 2
        self.sep_id = 3

        # Token → ID
        self.token_to_id = {
            PAD_TOKEN: self.pad_id,
            UNK_TOKEN: self.unk_id,
            CLS_TOKEN: self.cls_id,
            SEP_TOKEN: self.sep_id,
        }

        # ID → Token
        self.id_to_token = {
            self.pad_id: PAD_TOKEN,
            self.unk_id: UNK_TOKEN,
            self.cls_id: CLS_TOKEN,
            self.sep_id: SEP_TOKEN,
        }


    # =====================================================
    # Tokenization
    # =====================================================

    def tokenize(self, text):

        text = str(text).lower()

        tokens = text.split()

        return tokens


    # =====================================================
    # Build Vocabulary
    # =====================================================

    def build_vocab(self, texts):

        counter = Counter()

        for text in texts:

            tokens = self.tokenize(text)

            counter.update(tokens)

        # Special Token 4개를 제외한 vocabulary 개수
        max_words = self.vocab_size - 4

        most_common_tokens = counter.most_common(
            max_words
        )

        for token, _ in most_common_tokens:

            if token not in self.token_to_id:

                token_id = len(
                    self.token_to_id
                )

                self.token_to_id[token] = token_id

                self.id_to_token[token_id] = token

        print("Vocabulary")
        print("-" * 70)

        print(
            f"Vocabulary Size : "
            f"{len(self.token_to_id):,}"
        )

        print()


    # =====================================================
    # Encode
    # =====================================================

    def encode(
        self,
        text,
        max_length=100
    ):

        tokens = self.tokenize(text)

        tokens = tokens[:max_length - 2]

        # CLS + Text + SEP
        tokens = (
            [self.cls_token]
            + tokens
            + [self.sep_token]
        )

        # Maximum Length
        tokens = tokens[:max_length]

        input_ids = [
            self.token_to_id.get(
                token,
                self.unk_id
            )
            for token in tokens
        ]

        return input_ids


    # =====================================================
    # Save Vocabulary
    # =====================================================

    def save(self, path):

        import json

        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {
                str(key): value
                for key, value
                in self.id_to_token.items()
            }
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )