# import pytest
# import numpy as np
# import sys
# from pathlib import Path
# import os

# src_path = Path(__file__).parent.parent / "src"
# sys.path.insert(0, str(src_path))

# from topology_generation.topology_generation import (
#     determine_2d_patch_batch,
#     determine_3d_patch_batch,
#     determine_pooling_operations,
#     determine_channels_per_layer,
#     generate_network_topologies,
# )


# @pytest.mark.parametrize(
#     "dims, expected",
#     (
#         ((192, 160), (5, 5)),
#         ((128, 128, 128), (5, 5, 5)),
#         ((320, 256), (6, 6)),
#         ((80, 192, 128), (4, 5, 5)),
#         ((64, 160, 128), (4, 5, 5)),
#         ((512, 512), (6, 6)),
#         ((56, 40), (3, 3)),
#         ((40, 56, 40), (3, 3, 3)),
#         ((320, 320), (6, 6)),
#         ((20, 192, 192), (2, 5, 5)),
#         ((112, 128, 128), (4, 5, 5)),
#         ((96, 160, 128), (4, 5, 5)),
#     ),
# )
# def test_determine_pooling_ops(dims, expected):
#     result = determine_pooling_operations(dims)
#     assert result == expected

# @pytest.mark.parametrize(
#     "pooling_ops, expected", 
#     (
#         ((3, 3, 3), (32, 64, 128)),
#         ((3, 3), (32, 64, 128)),
#         ((5, 5, 5), (32, 64, 128, 256, 256)),
#         ((5, 5), (32, 64, 128, 256, 512)), 
#         ((4, 5, 5), (32, 64, 128, 256, 256)),
#         ((6, 6), (32, 64, 128, 256, 512, 512)),
#         ((2, 5, 5), (32, 64, 128, 256, 256))
#     )
# )
# def test_channels_per_layer(pooling_ops, expected):
#     result = determine_channels_per_layer(pooling_ops)
#     assert result == expected


# # @pytest.mark.parametrize(
# #     "dims, expected",
# #     (
# #         ((192, 160), (5, 5)),
# #         ((128, 128, 128), (5, 5, 5)),
# #         ((320, 256), (6, 6)),
# #         ((80, 192, 128), (4, 5, 5)),
# #         ((64, 160, 128), (4, 5, 5)),
# #         ((512, 512), (6, 6)),
# #         ((56, 40), (3, 3)),
# #         ((40, 56, 40), (3, 3, 3)),
# #         ((320, 320), (6, 6)),
# #         ((20, 192, 192), (2, 5, 5)),
# #         ((112, 128, 128), (4, 5, 5)),
# #         ((96, 160, 128), (4, 5, 5)),
# #     ),
# # )
# # def test_2d_patch_size(dims, expected):
# #     result = determine_2d_patch_batch(dims, spacing, )
# #     assert result == expected