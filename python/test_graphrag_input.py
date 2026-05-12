import asyncio
import logging
from pathlib import Path
from graphrag_input.input_config import InputConfig
from graphrag_input.text import TextFileReader
from graphrag_storage.factory import create_storage
from graphrag_storage.config import StorageConfig
from graphrag_input.input_type import InputType

logging.basicConfig(level=logging.DEBUG)

async def test_reader():
    storage_config = StorageConfig(type="file", base_dir="../../data/harusame-md")
    storage = create_storage(storage_config)
    
    input_config = InputConfig(
        type=InputType.Text,
        base_dir="../../data/harusame-md",
        file_pattern=".*\\.txt$"
    )
    
    reader = TextFileReader(storage=storage, **input_config.model_dump())
    
    try:
        data = await reader.read()
        print("Data loaded:", data)
    except Exception as e:
        print("Error reading:", e)

asyncio.run(test_reader())