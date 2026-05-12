import asyncio
import logging
from pathlib import Path
from graphrag_storage.file_storage import FileStorage
import re

async def main():
    base_dir = r"D:\Github\Fujinami\python\data\harusame-md"
    storage = FileStorage(base_dir=base_dir)
    pattern = re.compile(r".*\.txt$")
    files = list(storage.find(pattern))
    print("Files found:", files)

asyncio.run(main())