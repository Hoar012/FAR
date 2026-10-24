import multiprocessing as mp

from env_def import *
from mani_skill.trajectory.replay_trajectory import main as replay_main
from mani_skill.trajectory.replay_trajectory import parse_args


def main():
    mp.set_start_method("spawn")
    replay_main(parse_args())


if __name__ == "__main__":
    main()
