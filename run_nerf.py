
from xrnerf.core.apis import *


if __name__ == "__main__":

    args = parse_args()
    # cfg = Config.fromfile(args.config)
    run_nerf(args)
