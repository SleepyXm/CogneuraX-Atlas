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
    def test_every_chunk_enters_tree_and_facets_remain_separate(self):
        chunks = [SimpleNamespace(index=i, meta=SimpleNamespace(doc_items=[], headings=[], captions=[])) for i in range(3)]

        class Converter:
            paths = []

            def convert(self, path):
                self.paths.append(path)
                return SimpleNamespace(document=object())

        class Chunker:
            def chunk(self, _):
                return iter(chunks)

            def contextualize(self, chunk):
                return f"chunk-{chunk.index}"

        class Tokenizer:
            def encode(self, text, add_special_tokens=True):
                if text == "OVERVIEW: oversized":
                    return list(range(513))
                return list(range(2))

        class Summarizer:
            calls = []

            def get_response(self, query_str, text_chunks):
                self.calls.append(list(text_chunks))
                return "TOPIC: compact\nTOPIC: compact\nOVERVIEW: oversized\nPERSON: Alice"

        summarizer = Summarizer()
        components = (Converter(), Chunker(), Tokenizer(), summarizer, 512)
        with patch.object(main, "load_processing_components", return_value=components):
            result = main.route_document(UploadFile(filename="note.md", file=BytesIO(b"text")))

        self.assertEqual(summarizer.calls, [["chunk-0", "chunk-1", "chunk-2"]])
        self.assertEqual(result["routes"], [
            {"type": "filename", "text": "Filename: note.md"},
            {"type": "topic", "text": "TOPIC: compact"},
            {"type": "person", "text": "PERSON: Alice"},
        ])
        self.assertEqual(len(result["chunks"]), 3)

        with patch.object(main, "load_processing_components", return_value=components):
            main.route_document(UploadFile(filename="requirements.txt", file=BytesIO(b"package>=2,<3")))
        self.assertTrue(Converter.paths[-1].endswith(".md"))


if __name__ == "__main__":
    unittest.main()
