import sys
import os
import importlib

from omegaconf import OmegaConf
from tqdm.auto import tqdm

import torch

sys.path.append(os.path.join(os.path.dirname(__file__),'../..'))



def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def prepare_dataloader_for_rank(config, global_rank, num_processes=-1, repeat_cp_size=1):
    """ Get the dataloader given config and the current global rank.
        "dataset_setting" provides the list of dataset configs
        "rank_index_map" provides how to distribute the config across ranks
    """
    # repeat each elements in CP; [a b c] --> [a a ... b b ... c c ...]
    if repeat_cp_size > 1:
        print(f'before repeat config.rank_index_map: {config.rank_index_map}')
        repeated_rank_index_map = [element for element in config.rank_index_map for _ in range(repeat_cp_size)]
        config.rank_index_map = repeated_rank_index_map
        print(f'after repeat repeated_rank_index_map: {config.rank_index_map}')

    # get the dataset index
    num_total_indices = len(config.rank_index_map)
    dataset_index = config.rank_index_map[global_rank % num_total_indices]

    # get the correct partition
    num_partitions = 1
    partition_id = 0
    if num_processes > 0:
        rank_to_dataset_index_map = list(config.rank_index_map) * num_processes
        rank_to_dataset_index_map = rank_to_dataset_index_map[:num_processes]
        num_partitions = rank_to_dataset_index_map.count(dataset_index)
        partition_id = rank_to_dataset_index_map[:global_rank].count(dataset_index)
        print(f'rank_to_dataset_index_map: {rank_to_dataset_index_map}')
        print(f'dataset_index: {dataset_index} partition_id: {partition_id} num_partitions: {num_partitions} ')

    # get the loss weight scale factor to normalize loss weight to 1.0
    sum_loss_weight = 0.0
    for i in range(num_total_indices):
        dataset_setting = config.dataset_setting[config.rank_index_map[i]]
        sum_loss_weight += dataset_setting.get("loss_weight", 1.0)
    loss_weight_scale = float(num_total_indices) / sum_loss_weight

    # fetch the config
    dataset_setting = config.dataset_setting[dataset_index]
    loss_weight = dataset_setting.get("loss_weight", 1.0) * loss_weight_scale
    print(f'global_rank: {global_rank} -- dataset_index: {dataset_index} - loss_weight_scale: {loss_weight_scale} - loss weight: {loss_weight} - dataset_setting: {dataset_setting}')

    # set prompt function
    utils_prompt_module = importlib.import_module(dataset_setting.get_prompt_module)
    get_prompt_func = getattr(utils_prompt_module, dataset_setting.get_prompt_func)
    get_prompt_frame_spans_func = None
    if hasattr(dataset_setting, "get_prompt_frame_spans_func"):
        get_prompt_frame_spans_func = getattr(utils_prompt_module, dataset_setting.get_prompt_frame_spans_func)

    # get dataset from setting
    dataset_kwargs = dataset_setting.get("dataset_kwargs", dict())

    # get bucket configs
    assert hasattr(dataset_kwargs, "bucket_configs")
    bucket_configs = dataset_kwargs.get("bucket_configs", dict())

    dataset = get_obj_from_str(dataset_setting.dataset_target)(
        get_prompt_func=get_prompt_func,
        get_prompt_frame_spans_func=get_prompt_frame_spans_func,
        partition_id=partition_id,
        num_partitions=num_partitions,
        **dataset_kwargs
    )

    # get dataloader from setting
    dataloader_kwargs = dataset_setting.get("dataloader_kwargs", dict())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        **dataloader_kwargs,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        collate_fn = dataset.collate_fn if hasattr(dataset,"collate_fn") else None,
    )

    return dataloader, loss_weight, bucket_configs



if __name__ == '__main__':
    # example_config_path = 'source/dataset/example_config.yaml'
    example_config_path = "configs/train_t2v_opensora_v2_ms_long32_hq400.yaml"
    config = OmegaConf.load(example_config_path)

    dataloader = prepare_dataloader_for_rank(config.video_training_data_config, global_rank=7, num_processes=28)

    num_train_steps = 1000
    progress_bar = tqdm(range(0, num_train_steps))

    # output_dir = "assets/webvid-trimming_aes-tfreader"
    # os.makedirs(output_dir, exist_ok=True)

    # for step, batch in enumerate(tfreader):
    for step, batch in enumerate(dataloader):
        progress_bar.update(1)

        # # save data for visualization
        # pixel_values = batch['pixel_values'].cpu()
        # pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
        # for idx, pixel_value in enumerate(pixel_values):
        #     pixel_value = pixel_value[None, ...]
        #     text_value = batch['text'][idx]
        #     of_score = batch['of_score'][idx]
        #     fps_value = batch['fps'][idx]
        #     text_value = (text_value[:70] + '..') if len(text_value) > 70 else text_value
        #     output_filename = f"{output_dir}/{f'{fps_value}-{of_score}-{text_value}'}.gif"
        #     print(f'saving data to {output_filename}')
        #     save_videos_grid(pixel_value, output_filename, rescale=True)

        # print(f'step: {step} / num_train_steps: {num_train_steps}')

        if step >= num_train_steps:
            break
