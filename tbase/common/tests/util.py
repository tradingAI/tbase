def smoketest(argstr, **kwargs):
    import tempfile
    import subprocess
    import os
    argstr = 'python -m baselines.run ' + argstr
    for key, value in kwargs:
        argstr += ' --{}={}'.format(key, value)
    tempdir = tempfile.mkdtemp()
    env = os.environ.copy()
    env['OPENAI_LOGDIR'] = tempdir
    subprocess.run(argstr.split(' '), env=env)
    return tempdir
