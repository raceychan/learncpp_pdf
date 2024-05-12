# LearnCPP_PDF

## Disclaimer

All content directly comes from the [learncpp.com](https://learncpp.com) website, no content changed, some decorative elements and the comment section is removed for better readability.

> *Please consider support the website here [learncpp-about](https://www.learncpp.com/about/)*

since it specifically states that a pdf version should not be spread out by anyone and people should instead make pdf on their own, this tool is hence created.

## Usage

1.clone the repo

```bash
git clone 'git@github.com:raceychan/learncpp_pdf.git'
```

2.cd to src folder

```bash
cd learncpp_pdf
```

3.execute the application

```bash
make install && make run
```

### Configuration

You can create a '.env' file under the project root, the program will read them.

| key| type| default|
| --- | --- | --- |
| DOWNLOAD_CONCURRENT_MAX |int | 200 |
| COMPUTE_PROCESS_MAX | int | os.cpu_count() |
| PDF_CONVERTION_MAX_RETRY | int | 3 |
| BOOK_NAME | str | 'learncpp.pdf |
| REMOVE_CACHE_ON_SUCCESS | bool | False |

Note: setting DOWNLOAD_CONCURRENT_MAX to higher number might boost download speed, but some requests might fail as it exerts more pressure on the website

## CLI

You can use cli with following options to force-redo an action.

```bash
python -m book --help
```

```bash
options:
  -h, --help      show this help message and exit
  -D, --download  Downloading articles from learcpp.com, ignore cache
  -C, --convert   Converting downloaded htmls to pdfs, ignore cache
  -M, --merge     Merging Chapters into a single book, ignore cache
  -R, --rmcache   Remove the cache folder
  -A, --all       Download, convert and merge
```

example: re-run the convert process and remove the cache folder

```bash
python -m book --convert --rmcache
```

if not command specified, all actions will be taken(cache would be applied to avoid uncessary requests).

## Use-Tips

- It is possible that the download process and/or the convert process might fail due to various reason, for example, the target site is overloaded, in most cases, you can simply just re-run the program to solve these problems.
However, if you do think it is a bug, always feel free to post an issue.

- You might want to compress the pdf book for performance and storage.
check [pdfsizeopt](https://github.com/pts/pdfsizeopt) out 


## Features

- Ultra fast, utilize concurrency for scraping and parallel for making PDF, the whole process is expected to finish within a few minutes.
- Rich cli interface showing realtime progress of the application
- Cache on fail, you can just re-run the application without worrying about redundant IO or calcualtion.

## Alternatives

- [LearnCPP Downloader](https://github.com/amalrajan/learncpp-download/tree/master)

This does not utilize concurrent requests and multiprocessing, so it takes substantially more time to do the job.
