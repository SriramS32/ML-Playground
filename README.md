# Playground
Just contains scripts and projects in dev for now

Tensorflow and scraping scripts

## File Summaries
- `scrape_audio.py` Scrapes audio files from a KDFC (classical music) website using Python's multiprocessing for parallel programming to speed up downloads and `redis` as a key store to prevent redundant downloads. Requires `bs4` (an html/xml parsing library), `redis` (in-memory data structure store), and `urllib2`.

Style transfer of a google maps image of a location near me:
![House in Udnie Style](house_udnie.png)