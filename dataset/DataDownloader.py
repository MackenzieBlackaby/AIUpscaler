from kagglehub import dataset_download

# Downloads the data on the fly. Default flickr2k dataset
def DownloadData(name: str = "daehoyang/flickr2k"):
    path = dataset_download("daehoyang/flickr2k")
    return path
