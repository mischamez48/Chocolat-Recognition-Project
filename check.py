import argparse
import sys
from typing import Union

import pandas as pd

IDS = [
	1000757,
	1000758,
	1000759,
	1000760,
	1000761,
	1000762,
	1000764,
	1000766,
	1000767,
	1000769,
	1000770,
	1000773,
	1000774,
	1000775,
	1000776,
	1000777,
	1000778,
	1000781,
	1000782,
	1000783,
	1000784,
	1000786,
	1000789,
	1000790,
	1000794,
	1000795,
	1000796,
	1000798,
	1000800,
	1000801,
	1000806,
	1000807,
	1000809,
	1000811,
	1000813,
	1000814,
	1000816,
	1000818,
	1000819,
	1000822,
	1000823,
	1000824,
	1000825,
	1000829,
	1000830,
	1000832,
	1000833,
	1000834,
	1000835,
	1000837,
	1000839,
	1000840,
	1000841,
	1000842,
	1000845,
	1000846,
	1000847,
	1000848,
	1000852,
	1000853,
	1000856,
	1000857,
	1000858,
	1000861,
	1000862,
	1000863,
	1000864,
	1000867,
	1000869,
	1000871,
	1000873,
	1000874,
	1000877,
	1000878,
	1000879,
	1000881,
	1000883,
	1000884,
	1000886,
	1000887,
	1000891,
	1000892,
	1000893,
	1000895,
	1000897,
	1000898,
	1000901,
	1000902,
	1000904,
	1000905,
	1000906,
	1000907,
	1000908,
	1000911,
	1000912,
	1000913,
	1000915,
	1000917,
	1000918,
	1000919,
	1000921,
	1000924,
	1000925,
	1000927,
	1000929,
	1000931,
	1000933,
	1000934,
	1000935,
	1000936,
	1000937,
	1000938,
	1000940,
	1000942,
	1000943,
	1000944,
	1000945,
	1000947,
	1000948,
	1000949,
	1000955,
	1000956,
	1000958,
	1000959,
	1000960,
	1000961,
	1000962,
	1000966,
	1000967,
	1000968,
	1000969,
	1000970,
	1000973,
	1000975,
	1000976,
	1000978,
	1000980,
	1000982,
	1000983,
	1000984,
	1000986,
	1000988,
	1000990,
	1000991,
	1000994,
	1000995,
	1000996,
	1000997,
	1000998,
	1000999,
	1010001,
	1010002,
	1010003,
	1010005,
	1010006,
	1010007,
	1010010,
	1010011,
	1010016,
	1010018,
	1010020,
	1010021,
	1010023,
	1010024,
	1010025,
	1010027,
	1010028,
	1010030,
	1010033,
	1010034,
	1010036,
	1010037,
	1010038,
	1010039,
	1010040,
	1010042,
	1010043,
	1010044,
	1010046,
	1010047
]
COLS = [
	"Jelly White",
	"Jelly Milk",
	"Jelly Black",
	"Amandina",
	"Crème brulée",
	"Triangolo",
	"Tentation noir",
	"Comtesse",
	"Noblesse",
	"Noir authentique",
	"Passion au lait",
	"Stracciatella",
	"Arabia"
]


class TermFormat:
	RESET = '\033[0m'
	##
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	##
	BLUE = '\033[94m'
	CYAN = '\033[96m'
	GREEN = '\033[92m'
	PURPLE = '\033[95m'
	RED = '\033[91m'
	YELLOW = '\033[93m'
	
	@classmethod
	def format(cls, s: str, c: str):
		s = f"{c}{s}{cls.RESET}"
		return s


def check_df(df: pd.DataFrame, df_name: Union[str, None] = None):
	##
	miss_cols = set(COLS + ["id"]) - set(df.columns)
	if len(miss_cols):
		print(TermFormat.format(f"Missing columns: {list(miss_cols)}", TermFormat.RED))
		sys.exit()
	##
	df = df.set_index("id")
	err_strs = []
	##
	add_cols = set(df.columns) - set(COLS)
	if len(add_cols):
		err_strs.append(TermFormat.format(f"Additional columns: {list(add_cols)}", TermFormat.RED))
	##
	miss_ids = set(IDS) - set(df.index)
	if len(miss_ids):
		err_strs.append(TermFormat.format(f"Missing IDs: {list(miss_ids)}", TermFormat.RED))
	##
	add_ids = set(df.index) - set(IDS)
	if len(add_ids):
		err_strs.append(TermFormat.format(f"Additional IDs: {list(add_ids)}", TermFormat.RED))
	##
	non_int_cols = df.dtypes[df.dtypes != int]
	if len(non_int_cols):
		err_strs.append(TermFormat.format(f"Non-int columns: {non_int_cols.index.tolist()}", TermFormat.RED))
	##
	neg_rows_mask = (df < 0).any(axis=1)
	neg_rows_mask = neg_rows_mask[neg_rows_mask == True]
	if len(neg_rows_mask):
		err_strs.append(TermFormat.format(f"IDs with negative values: {neg_rows_mask.index.tolist()}", TermFormat.RED))
	##
	if len(err_strs):
		if df_name is not None:
			err_strs.insert(0, TermFormat.format(df_name, TermFormat.BOLD + TermFormat.UNDERLINE))
		print('\n'.join(err_strs))
		sys.exit()
	return df


def check_submission(sub_path: str):
	df_sub = pd.read_csv(sub_path, index_col=None)
	df_sub = check_df(df_sub)
	print(TermFormat.format("OK", TermFormat.GREEN))


def match_submission(local_sub_path: str, kaggle_sub_path: str):
	df_local_sub = pd.read_csv(local_sub_path, index_col=None)
	df_local_sub = check_df(df_local_sub, df_name="local")
	df_kaggle_sub = pd.read_csv(kaggle_sub_path, index_col=None)
	df_kaggle_sub = check_df(df_kaggle_sub, df_name="kaggle")
	df_local_sub = df_local_sub.reindex(IDS, axis=0).reindex(COLS, axis=1)
	df_kaggle_sub = df_kaggle_sub.reindex(IDS, axis=0).reindex(COLS, axis=1)
	comp = df_local_sub.compare(df_kaggle_sub, align_axis=0, result_names=("local", "kaggle"))
	num_diff = comp.notna().to_numpy().sum() // 2
	if num_diff == 0:
		print(TermFormat.format("0 differences found.", TermFormat.GREEN))
	else:
		comp = df_local_sub.compare(df_kaggle_sub, align_axis=0, result_names=("local", "kaggle"))
		print(TermFormat.format(f"{num_diff} differences found (NaN = no difference):", TermFormat.RED))
		print(TermFormat.format(comp, TermFormat.RED))


if __name__ == "__main__":
	main_parser = argparse.ArgumentParser()
	subparsers = main_parser.add_subparsers(required=True, dest="command")
	check_parser = subparsers.add_parser("check", help="checks whether the submission file is valid")
	check_parser.add_argument("--path", type=str, default="submission.csv", help="path of submission file to check (default: 'submission.csv')")
	match_parser = subparsers.add_parser("match", help="checks whether two submission files match")
	match_parser.add_argument("--local", type=str, default="submission.csv", help="path of local submission file (default: 'submission.csv')")
	match_parser.add_argument("--kaggle", type=str, default="kaggle.csv", help="path of Kaggle submission file (default: 'kaggle.csv')")
	args = main_parser.parse_args()
	if args.command == "check":
		check_submission(args.path)
	else:  # "match"
		match_submission(args.local, args.kaggle)
