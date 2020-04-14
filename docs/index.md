## Quick start

### Dependency
- Mac OS/Linux
- python 3.5-3.7
- pip

### Linux/macOS Install
Python 3.5–3.7, pip
```
pip install tbase>=0.1.2
export TUSHARE_TOKEN=YOUR_TOKEN
```
Reference: [Tushare token register](https://tushare.pro/register?reg=124861)
### [Docker guide](docker_guide.md)

### Usage examples
- ddpg: `python -m tbase.run --alg=ddpg --codes=000001.SZ`
- td3: `python -m tbase.run --alg=td3 --codes=000001.SZ`

## Questions
- [How to reproducible?](reproducible.md)
