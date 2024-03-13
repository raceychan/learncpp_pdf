# LearnCPP_PDF

## Disclaimer

All content comes from [learncpp.com](https://learncpp.com), since it specifically states that pdf version should be spread out and people should instead make pdf out of their own, hence this tool

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

## Features

- Ultra fast, utilize concurrency for scraping and parallel for making PDF, the whole process is expected to finish within a few minutes.
- Rich cli interface showing realtime progress of the application
- Cached on fail, you can just re-run the application without worrying about redundant IO or calcualtion.


## Alternatives

- [LearnCPP Downloader](https://github.com/amalrajan/learncpp-download/tree/master)

This does not utilize concurrent requests and multiprocessing, so it takes substantially more time to do the job.