## Quick start

### Dependency
- python 3.5-3.7
- pip

### Install
```
pip install tbase>=0.1.2
export TUSHARE_TOKEN=YOUR_TOKEN
```
Reference: [Tushare token register](https://tushare.pro/register?reg=124861)

### Usage examples
- ddpg: `python -m tbase.run --alg=ddpg --codes='000001.SZ'`
- td3: `python -m tbase.run --alg=td3 --codes='000001.SZ'`


## Questions
- [How to reproducible?](reproducible.md)
