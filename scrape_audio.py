from argparse import ArgumentParser
from bs4 import BeautifulSoup
from multiprocessing import Pool
from time import time
from urllib2 import urlopen
import os, sys

links = ["http://www.kdfc.com/series/kdfcs-the-state-of-the-arts/"]

def parse_data(args):
	video_sublinks = []
	processes = args.processes if args.processes is not None else 8
	directory = os.path.join("radio", "KDFC")
	if not os.path.exists(directory):
		os.makedirs(directory)

	for site_url in links:
		u = urlopen(site_url)
		try:
			html = u.read().decode('utf-8')
		except:
			pass
		soup = BeautifulSoup(html, 'html5lib')
		for div in soup.find_all('div', attrs={"class":"podcast-player"}):
			for src in [link.get('src') for link in div.find_all('source')]:
				audio_file_name = os.path.join(directory, src.split('/')[-1])
				video_sublinks.append([src, audio_file_name])
	start = int(round(time()))
	pool = Pool(processes=processes)
	pool.map(download_audio, video_sublinks)
	print("Downloading took: {}s".format(int(round(time()))-start))
	start = int(round(time()))

def download_audio(audio_entry):
	audio_link = audio_entry[0]
	file_name = audio_entry[1]

	u = urlopen(audio_link)
	f = open(file_name, 'wb')
	# file_size = int(meta.getheaders("Content-Length")[0])
	# print("Downloading: %s Bytes: %s" % (file_name, file_size))
	block_sz = 8192
	while True:
		buffer = u.read(block_sz)
		if not buffer:
			break
		f.write(buffer)

if __name__ == '__main__':
	parser = ArgumentParser('Parse links.')
	parser.add_argument('--processes', type=int, help='number of processes to run parallelize')
	args = parser.parse_args()
	parse_data(args)
