import argparse as ap
import logging
import sys

import numpy as np
import pandas as pd

from jax.config import config

from jaxqtl.families.distribution import Gaussian, NegativeBinomial, Poisson
from jaxqtl.sim import run_sim


def get_logger(name, path=None):
    """get logger for factorgo progress"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = 0
        console = logging.StreamHandler()
        logger.addHandler(console)

        # if need millisecond use : %(asctime)s.%(msecs)03d
        log_format = "[%(asctime)s - %(levelname)s] %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        console.setFormatter(formatter)

        if path is not None:
            disk_log_stream = open("{}.log".format(path), "w")
            disk_handler = logging.StreamHandler(disk_log_stream)
            logger.addHandler(disk_handler)
            disk_handler.setFormatter(formatter)

    return logger


def main(args):
    argp = ap.ArgumentParser(description="")  # create an instance
    argp.add_argument(
        "-true-model", type=str, choices=["gaussian", "poisson", "NB"], help="Model"
    )
    argp.add_argument(
        "-null-alt", type=str, choices=["null", "alt"], help="Cis or nominal mapping"
    )
    argp.add_argument("-seed", type=int, default=1)
    argp.add_argument("-nsim", type=int, default=1000)
    argp.add_argument("-alpha", type=float, default=0.01)
    argp.add_argument("-maf", type=float, default=0.3)
    argp.add_argument("-true-beta", type=float, default=0.0)
    argp.add_argument("-beta0", type=float, default=0.0)
    argp.add_argument("-fwer", type=float, default=0.05)
    argp.add_argument(
        "-threshold",
        type=float,
        default=0.0,
        help="threshold for express percentage to run jaxqtl",
    )
    argp.add_argument("-sampleN", type=int, default=1000, help="sample size")
    argp.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose for logger",
    )
    argp.add_argument("-out", type=str, help="out file prefix")

    args = argp.parse_args(args)  # a list a strings

    platform = "cpu"
    config.update("jax_enable_x64", True)
    config.update("jax_platform_name", platform)

    log = get_logger(__name__, args.out)
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    if args.true_model == "NB":
        sim_family = NegativeBinomial()
    elif args.true_model == "poisson":
        sim_family = Poisson()
    elif args.true_model == "gaussian":
        sim_family = Gaussian()
    else:
        log.info("Please choose the available family")
        sys.exit()

    fwer = args.fwer

    res = run_sim(
        seed=args.seed,
        alpha=args.alpha,
        maf=args.maf,
        n=args.sampleN,
        model=args.null_alt,
        num_sim=args.nsim,
        true_beta=args.true_beta,
        sim_family=sim_family,
        beta0=args.beta0,
        jaxqtl_threshold=args.threshold,
    )

    d = {
        "rej_nb_wald": [np.mean(res.pval_nb_wald[~np.isnan(res.pval_nb_wald)] < fwer)],
        "rej_pois_wald": [
            np.mean(res.pval_pois_wald[~np.isnan(res.pval_pois_wald)] < fwer)
        ],
        "rej_nb_wald_robust": [
            np.mean(res.pval_nb_wald_robust[~np.isnan(res.pval_nb_wald_robust)] < fwer)
        ],
        "rej_pois_wald_robust": [
            np.mean(
                res.pval_pois_wald_robust[~np.isnan(res.pval_pois_wald_robust)] < fwer
            )
        ],
        "rej_lm": [np.mean(res.pval_lm[~np.isnan(res.pval_lm)] < fwer)],
        "rej_nb_score": [
            np.mean(res.pval_nb_score[~np.isnan(res.pval_nb_score)] < fwer)
        ],
        "rej_pois_score": [
            np.mean(res.pval_pois_score[~np.isnan(res.pval_pois_score)] < fwer)
        ],
        "rej_lm_gtex": [np.mean(res.pval_lm_gtex[~np.isnan(res.pval_lm_gtex)] < fwer)],
        "rej_nb_score_thr": [
            np.mean(res.pval_nb_score_thr[~np.isnan(res.pval_nb_score_thr)] < fwer)
        ],
    }

    df_rej = pd.DataFrame(data=d)
    df_rej.to_csv(args.out + ".tsv", sep="\t", index=False)

    d = {
        "rej_nb_wald": res.pval_nb_wald,
        "rej_pois_wald": res.pval_pois_wald,
        "rej_nb_wald_robust": res.pval_nb_wald_robust,
        "rej_pois_wald_robust": res.pval_pois_wald_robust,
        "rej_lm": res.pval_lm,
        "rej_nb_score": res.pval_nb_score,
        "rej_pois_score": res.pval_pois_score,
        "rej_lm_gtex": res.pval_lm_gtex,
        "rej_nb_score_thr": res.pval_nb_score_thr,
    }

    df_pval = pd.DataFrame(data=d)
    df_pval.to_csv(args.out + ".pval.tsv", sep="\t", index=False)

    return 0


# user call this script will treat it like a program
if __name__ == "__main__":
    sys.exit(
        main(sys.argv[1:])
    )  # grab all arguments; first arg is alway the name of the script
