from argparse import ArgumentParser
from bs4 import BeautifulSoup
from multiprocessing import Pool
from redis import StrictRedis
from time import time
from urllib2 import urlopen, URLError
import os, sys

links = ["http://www.kdfc.com/series/kdfcs-the-state-of-the-arts/"]
redis = StrictRedis(host='localhost', port=6379, db=0)

def parse_data(args):
	""" Parses audio links from radio websites using bs4 and multiprocessing.

	Collates audio download links from parsing radio website html and then
	uses multiprocessing to parallelize downloads from those links.

	Args:
		args: cl arguments through argument parser
	"""
	video_sublinks = []
	processes = args.processes if args.processes is not None else 8
	directory = os.path.join("radio", "KDFC")
	if not os.path.exists(directory):
		os.makedirs(directory)

	for site_url in links:
		try:
			html = urlopen(site_url).read().decode('utf-8')
		except URLError, e:
			print("URLError: {} for site: {}".format(str(e.reason), site_url))
			continue
		soup = BeautifulSoup(html, 'html5lib')
		for div in soup.find_all('div', attrs={"class":"podcast-player"}):
			for src in [link.get('src') for link in div.find_all('source')]:
				audio_file_name = os.path.join(directory, src.split('/')[-1])
				# Append both link and name to one list entry so pool.map()
				# can take one list parameter
				video_sublinks.append([src, audio_file_name])
	start = int(round(time()))
	pool = Pool(processes=processes)
	pool.map(download_audio, video_sublinks)
	print("Downloading took: {}s".format(int(round(time()))-start))
	start = int(round(time()))

def download_audio(audio_entry):
	""" Downloads data from audio links using a buffer.
	
	Args:
		audio_entry: a list containing two entries. First entry is the audio source url.
					Second entry is file name to save the audio data to.
	"""
	# Unwrapping audio entry
	audio_link = audio_entry[0]
	file_name = audio_entry[1]
	if redis.exists(file_name):
		return
	try:
		u = urlopen(audio_link)
	except URLError, e:
		print("URLError: {} for site: {}".format(str(e.reason), audio_link))
		return

	f = open(file_name, 'wb')
	# file_size = int(meta.getheaders("Content-Length")[0])
	# print("Downloading: %s Bytes: %s" % (file_name, file_size))
	block_sz = 8192
	while True:
		buffer = u.read(block_sz)
		if not buffer:
			break
		f.write(buffer)
	redis.set(file_name, 1)

if __name__ == '__main__':
	parser = ArgumentParser('Parse links.')
	parser.add_argument('--processes', type=int, help='number of processes to run parallelize')
	parser.add_argument('--resetdb', action='store_true', help='Can be called to flush the database. Shutting down the server will still save the data (flushing will delete).')
	args = parser.parse_args()
	if args.resetdb:
		redis.flushdb() # Can be called to flush the database. Shutting down the server will still save the data (flushing will delete).
	parse_data(args)
