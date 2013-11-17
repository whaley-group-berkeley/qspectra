Viewing Examples
----------------

First, make sure you have IPython version 1.0 or later installed. Then, to view
these examples, navigate to this directory and run:
```
ipython notebook --pylab=inline
```

Eventually, we'll turn these into HTML (very easy with `ipython nbconvert`) and
host them online.

Testing Examples
----------------

To test reproducing these examples from the command line, download
`ipnbdoctest.py` from [this gist](https://gist.github.com/shoyer/7497853) and
add it to your path, i.e.,
```
git clone https://gist.github.com/7497853.git
cd 7497853
chmod +x ipnbdoctest.py
ln -s ipnbdoctest.py /usr/local/bin/ipnbdoctest
```

You'll also need to install [pypng](https://github.com/drj11/pypng):
```
pip install pypng
```

Then navigate to this directory and run:
```
ipnbdoctest *.ipynb
```

**Note**: This script has a time-out of 60 seconds for evaluating each cell. It
is also unclear whether or not graphics produced on different platforms (e.g.,
OS X vs Linux) are exactly identical, pixel for pixel. But this still is a
pretty neat way to do testing.
