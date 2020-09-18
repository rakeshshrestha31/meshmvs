import argparse
import json
import os


category_id_to_name = {
    '02828884': 'bench', '03001627': 'chair', '03636649': 'lamp', '03691459': 'speaker', '04090263': 'firearm',
    '04379243': 'table', '04530566': 'watercraft', '02691156': 'plane', '02933112': 'cabinet',
    '02958343': 'car', '03211117': 'monitor', '04256520': 'couch', '04401088': 'cellphone'
}

def parse_args():
    parser = argparse.ArgumentParser(description='extract category split files for generalization experiments.')
    parser.add_argument('splits_file', type=str)
    return parser.parse_args()


def parse_splits_file(splits_file):
    with open(splits_file, 'r') as f:
        splits = json.load(f)
        return splits


def generate_splits_files(splits, splits_dir, base_filename):
    # splits excluding a category
    excluding_splits = {
        category_id: {split_id: {} for split_id in ['train', 'val', 'test']}
        for category_id in category_id_to_name.keys()
    }
    # splits including only a category
    including_only_splits = {
        category_id: {split_id: {} for split_id in ['train', 'val', 'test']}
        for category_id in category_id_to_name.keys()
    }

    for split_id in splits.keys():
        for category_id in splits[split_id].keys():
            including_only_splits[category_id][split_id][category_id] \
                = splits[split_id][category_id]
            for excluding_category_id in excluding_splits.keys():
                if excluding_category_id != category_id:
                    excluding_splits[excluding_category_id][split_id][category_id] \
                        = splits[split_id][category_id]

    for category_id, category_name in category_id_to_name.items():
        # excluding
        filename = '{}_excluding_{}.json'.format(base_filename, category_name)
        filepath = os.path.join(splits_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(excluding_splits[category_id], f)

        # including only
        filename = '{}_including_only_{}.json'.format(base_filename, category_name)
        filepath = os.path.join(splits_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(including_only_splits[category_id], f)

    sanity_check_splits(
        splits, excluding_splits, including_only_splits
    )


def sanity_check_splits(
        splits, excluding_splits, including_only_splits
):
    for split_id in splits.keys():
        for category_id in splits[split_id].keys():
            print(split_id, category_id_to_name[category_id], category_id)
            print(
                'original', len(splits[split_id][category_id])
            )
            if category_id in including_only_splits[category_id][split_id]:
                print(
                    'including_only',
                    {
                        key: len(val)
                        for key, val in \
                            including_only_splits[category_id][split_id].items()
                    }
                )
            print(
                'excluding_splits',
                {
                    key: len(val)
                    for key, val in \
                        excluding_splits[category_id][split_id].items()
                }
            )
            if category_id in excluding_splits[category_id][split_id]:
                print('excluding splits messed up for', category_id)
            print('\n\n')


if __name__ == '__main__':
    args = parse_args()
    splits_dir = os.path.dirname(args.splits_file)
    splits_filename_only = os.path.split(args.splits_file)[1]
    splits_basefilename = os.path.splitext(splits_filename_only)[0]
    splits = parse_splits_file(args.splits_file)
    generate_splits_files(splits, splits_dir, splits_basefilename)
