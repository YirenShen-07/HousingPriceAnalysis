"""Fetch the California census housing example used in Hands-On ML."""
from pathlib import Path
from urllib.request import urlopen
URL = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv'
PATH = Path(__file__).parent / 'data' / 'housing.csv'
def download():
    PATH.parent.mkdir(exist_ok=True)
    if not PATH.exists():
        with urlopen(URL,timeout=60) as response:
            PATH.write_bytes(response.read())
    return PATH
if __name__ == '__main__':
    print(download())
