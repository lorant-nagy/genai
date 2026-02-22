import yaml



# # this infers into params from config
# def infer_generic(config, params):
#     for key in config.generic.to_dict():
#         if hasattr(params, key):
#             setattr(params, key, getattr(config.generic, key))