import os
import platform
import re
from datetime import datetime
from email.message import Message
from email.parser import BytesParser
from typing import List, Union

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

from ofstedai.models.file import Chunk, File



def creation_date(path_to_file) -> datetime:
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.

    Args:
        path_to_file (str): The path to the file to be checked.
    
    Returns:
        datetime: The creation date of the file.
    """

    if platform.system() == "Windows":
        return datetime.utcfromtimestamp(os.path.getctime(path_to_file))
    else:
        stat = os.stat(path_to_file)
        try:
            return datetime.utcfromtimestamp(stat.st_birthtime)
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return datetime.utcfromtimestamp(stat.st_mtime)


def other_chunker(file: File, creator_user_uuid: str = "dev") -> List[Chunk]:
    elements = partition(filename=file.path)
    raw_chunks = chunk_by_title(elements=elements)

    chunks = []
    for i, raw_chunk in enumerate(raw_chunks):
        raw_chunk = raw_chunk.to_dict()
        raw_chunk["metadata"]["parent_doc_uuid"] = file.uuid

        chunk = Chunk(
            parent_file_uuid=file.uuid,
            index=i,
            text=raw_chunk["text"],
            metadata=raw_chunk["metadata"],
            creator_user_uuid=creator_user_uuid,
        )
        chunks.append(chunk)

    return chunks
