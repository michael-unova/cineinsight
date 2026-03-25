import urllib.request, zipfile

url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
print("Downloading MovieLens...")
urllib.request.urlretrieve(url, "data/movielens.zip")

with zipfile.ZipFile("data/movielens.zip", 'r') as z:
    z.extractall("data/")

print("Done! Data saved to data/ml-latest-small/")