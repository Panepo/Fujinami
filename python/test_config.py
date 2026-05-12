import asyncio
from pathlib import Path
from graphrag.config.load_config import load_config
import os

async def main():
    root = Path('ragdata/harusame-md')
    try:
        config = load_config(root_dir=root)
        print("BASE DIR:", config.input.base_dir)
        print("FILE PATTERN:", config.input.file_pattern)
        print("TYPE:", config.input.type)
        print("INPUT STORAGE:", config.input_storage)
    except Exception as e:
        print("Error loading config:", e)

asyncio.run(main())