import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import UploadFile

import main


class Values(list):
    def tolist(self):
        return list(self)


class ProcessorTest(unittest.TestCase):
    def test_every_docling_region_is_returned_in_order(self):
        chunks = [SimpleNamespace(index=i, meta=SimpleNamespace(doc_items=[], headings=[], captions=[])) for i in range(3)]

        class Converter:
            paths = []

            def convert(self, path):
                self.paths.append(path)
                return SimpleNamespace(document=object())

        class Chunker:
            class Tokenizer:
                @staticmethod
                def count_tokens(text):
                    return len(text.split())

                @staticmethod
                def get_max_tokens():
                    return 2

            tokenizer = Tokenizer()

            def chunk(self, _):
                return iter(chunks)

            def contextualize(self, chunk):
                return f"chunk-{chunk.index}"

        components = (Converter(), Chunker())
        with patch.object(main, "load_processing_components", return_value=components):
            result = main.process_document(UploadFile(filename="note.md", file=BytesIO(b"text")))

        self.assertEqual(result["regions"], [
            {"text": "chunk-0", "page": None, "region_index": 0},
            {"text": "chunk-1", "page": None, "region_index": 1},
            {"text": "chunk-2", "page": None, "region_index": 2},
        ])

        with patch.object(main, "load_processing_components", return_value=components):
            main.process_document(UploadFile(filename="requirements.txt", file=BytesIO(b"package>=2,<3")))
        self.assertTrue(Converter.paths[-1].endswith(".md"))

    def test_region_over_embedding_limit_is_rejected(self):
        chunk = SimpleNamespace(meta=SimpleNamespace(doc_items=[]))
        converter = SimpleNamespace(convert=lambda _: SimpleNamespace(document=object()))
        chunker = SimpleNamespace(
            chunk=lambda _: iter([chunk]),
            contextualize=lambda chunk: "one two three",
            tokenizer=SimpleNamespace(count_tokens=lambda text: len(text.split()), get_max_tokens=lambda: 2),
        )
        with patch.object(main, "load_processing_components", return_value=(converter, chunker)):
            with self.assertRaisesRegex(ValueError, "embedding token limit"):
                main.process_document(UploadFile(filename="note.md", file=BytesIO(b"text")))


if __name__ == "__main__":
    unittest.main()
