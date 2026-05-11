import os
import uuid
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from rag_example.core.s3_module import IS3Client
from rag_example.core.utils.text_cleaners import TextCleanerProtocol


def fetch_and_clean_text(url: str, cleaner: TextCleanerProtocol) -> str:
    text = []
    response = requests.get(url, headers={"User-Agent": "StudyRagAgent/1.0"})
    soup = BeautifulSoup(response.content, "html.parser")
    for paragraph in soup.find_all("p"):
        text.append(cleaner.normalize_text(paragraph.text))
    return "\n".join(text)


def save_articles_to_s3(s3_client: IS3Client, cleaner: TextCleanerProtocol, urls: list[str]) -> str:
    file_key = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.txt"
    with open(file_key, "w") as file:
        for url in urls:
            content = fetch_and_clean_text(url, cleaner)
            file.write(content + "\n")
    s3_client.upload_file(file_key)
    os.remove(file_key)
    return file_key
