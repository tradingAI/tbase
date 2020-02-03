# -*- coding:utf-8 -*-
import multiprocessing
import sys
from importlib import import_module

from tbase.common.cmd_util import common_arg_parser, logger, make_trade_env

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def get_default_network(env_type):
    return 'lstm'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # try to import the alg module from tbase
        alg_module = import_module('.'.join(['tbase', alg, submodule]))
        return alg_module
    except ImportError:
        raise("get_alg_module error: not import_module %s" % alg)


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def train(args, extra_args, env_type):
    print('env_type: {}'.format(args.env))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(
        args.alg, env.args, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin':
        ncpu = ncpu // 2
    nenv = args.num_env if args.num_env else ncpu
    alg = args.alg
    seed = args.seed
    env_type, env_id = get_env_type(args)

    env = make_vec_env()
    return env


def main(args):
    arg_parser = common_arg_parser()
    args = arg_parser.parse_known_args(args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
    else:
        rank = MPI.COMM_WORLD.Get_rank()

    model, env = train(args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play:
        logger.INFO("Running trained model")
        obs = env.reset()

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0

    env.close()

    return model


if __name__ == '__main__':
    main(sys.argv)
