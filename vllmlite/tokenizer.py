"""Byte-level tokenizer so the server runs with zero downloads.

One token per UTF-8 byte plus a few specials. Real BPE lives behind the same
interface via HFTokenizer when transformers happens to be installed.
"""

from __future__ import annotations

BOS = 256
EOS = 257
PAD = 258
VOCAB_SIZE = 261


class ByteTokenizer:
    bos_id = BOS
    eos_id = EOS
    pad_id = PAD
    vocab_size = VOCAB_SIZE
    name = "byte"

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        return [BOS] + ids if add_bos else ids

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def decode_stream(self, ids: list[int], already: int) -> tuple[str, int]:
        """Decode ids[already:] but hold back a dangling multi byte character.

        Returns (text_delta, new_already).
        """
        raw = bytes(i for i in ids[already:] if i < 256)
        if not raw:
            return "", len(ids)
        # walk back over continuation bytes so we never emit a replacement char
        # for a sequence that just hasn't finished arriving
        cut = len(raw)
        for back in range(1, min(4, len(raw)) + 1):
            b = raw[-back]
            if b < 0x80:
                break
            if b >= 0xC0:
                need = 2 if b < 0xE0 else 3 if b < 0xF0 else 4
                if back < need:
                    cut = len(raw) - back
                break
        text = raw[:cut].decode("utf-8", errors="replace")
        # map the byte count we consumed back to a token count
        consumed = 0
        used = 0
        for tid in ids[already:]:
            if consumed >= cut:
                break
            if tid < 256:
                consumed += 1
            used += 1
        return text, already + used


class HFTokenizer:
    name = "hf"

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tok)
        self.eos_id = self.tok.eos_token_id
        self.bos_id = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.eos_id
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_id

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        return self.tok.encode(text, add_special_tokens=add_bos)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def decode_stream(self, ids: list[int], already: int) -> tuple[str, int]:
        prev = self.tok.decode(ids[:already], skip_special_tokens=True) if already else ""
        full = self.tok.decode(ids, skip_special_tokens=True)
        if full.endswith("�"):
            return "", already
        return full[len(prev) :], len(ids)


def get_tokenizer(name: str | None = None):
    if not name or name == "byte":
        return ByteTokenizer()
    return HFTokenizer(name)
