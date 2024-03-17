# LearnCPP_PDF

## Disclaimer

All content comes from [learncpp.com](https://learncpp.com), since it specifically states that a pdf version should not be spread out by anyone and people should instead make pdf on their own, hence this tool is created

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

## CLI

You can use cli with following options
```bash
python -m book --help
```

```bash
options:
  -h, --help      show this help message and exit
  -D, --download  Downloading articles from learcpp.com
  -C, --convert   Converting downloaded htmls to pdfs
  -M, --merge     Merging Chapters into a single book
  -A, --all       Download, convert and merge
```

## Features

- Ultra fast, utilize concurrency for scraping and parallel for making PDF, the whole process is expected to finish within a few minutes.
- Rich cli interface showing realtime progress of the application
- Cache on fail, you can just re-run the application without worrying about redundant IO or calcualtion.

## Alternatives

- [LearnCPP Downloader](https://github.com/amalrajan/learncpp-download/tree/master)

This does not utilize concurrent requests and multiprocessing, so it takes substantially more time to do the job.
